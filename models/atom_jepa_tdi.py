"""Direction D: a head whose shape encodes what the TDI label actually means.

The classification task asks whether a molecule is a time-dependent inhibitor of CYP2D6 or
CYP3A4. `data/raw/cyp-challenge-TRAIN_TDI.csv` — the very file `prepare_data.py` reads and
then throws most of the columns away — also carries, for all four enzymes, the pIC50
measured under the TDI pre-incubation condition and the pIC50 measured directly. Measured
on the training rows alone, the difference between the two predicts the label almost
perfectly:

    shift = pIC50_TDI_condition - pIC50_direct_inhibition

    CYP2D6   AUC 0.962 (n=1493), best threshold 0.30 -> MCC 0.898, 96.1% agreement
    CYP3A4   AUC 0.855 (n=2334), best threshold 0.30 -> MCC 0.687, 82.0% agreement

The threshold lands on 0.30 for both enzymes independently, which is the textbook two-fold
shift in IC50. So `is_TDI` is, to a very good approximation, a *derived* quantity, and a
binary classifier is being asked to rediscover a subtraction and a comparison from scratch.

This model instead predicts the eight pIC50s and builds each logit out of them:

    logit_e = softplus(scale_e) * (p_TDI_e - p_direct_e - 0.30/sigma) + bias_e

`scale_e` and `bias_e` are learned, and `bias_e` starts at zero so the decision boundary
begins exactly at the threshold the training data implies and is then free to move. The
eight pIC50s get their own auxiliary MSE, which is what turns ~3.4k binary labels into
~3.4k x 8 real-valued ones.

Two variants, selected by the task:

  classification (`tdi_head=true`)  the structured logit above. All eight slots are
      standardized with ONE shared mean and std, because a difference of two differently
      scaled quantities has no physical meaning; the threshold then scales as 0.30/sigma.
  regression (`tdi_head=false`)     plain multi-task learning. Slots 0..3 are the four
      direct pIC50s the task is scored on and are left in the space finetune.py's own
      per-target standardization expects; slots 4..7 carry the TDI-condition pIC50s as
      auxiliary targets only, standardized per slot. The question here is narrower: does
      predicting the pre-incubation assay as a side task help the direct one?

The honest caveat, stated before the first run: the shift is a difference of two quantities
this encoder predicts poorly on its own — frozen features reach only R^2 ~ 0.15 on CYP2D6
direct pIC50 — and the difference of two noisy estimates can easily be noisier than either.
The design is principled; whether it pays is an empirical question, and the controls are
`AC_attn4_bce` (same features, same pooling, unstructured 2-wide head) for classification
and `AC_attn4_softthr` for regression.

Why this is not leakage, in the terms the project's rules use:

  * the auxiliary columns are *targets*, never inputs — nothing is fed to the encoder;
  * they are attached only for molecules in the TRAINING split, recorded by
    `fit_preprocessing(train)`; every other row carries NaN and is masked out, so the
    auxiliary loss is structurally incapable of seeing a validation or test label;
  * the 0.30 threshold is measured on training rows only;
  * the splits are untouched, and no row is added or removed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.atom_jepa_cached import CachedAtomJEPAModel, PooledSet, AtomSet, collate_atoms
from models.targets import with_interval

ENZYMES = ['CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']
DIRECT = [f'{e}_pIC50_direct_inhibition' for e in ENZYMES]
TDI = [f'{e}_pIC50_TDI_condition' for e in ENZYMES]
SLOTS = DIRECT + TDI                     # the eight head outputs, in this order
TDI_SLOTS = list(range(len(DIRECT), len(SLOTS)))
RAW = 'raw/cyp-challenge-TRAIN_TDI.csv'
# Which enzyme each classification target refers to, so the logits are built from the right
# pair of slots. The task's own target order is read from the loader, not assumed.
TDI_LABEL = {f'{e}_is_TDI': e for e in ENZYMES}
# Only two enzymes carry an is_TDI label in this challenge.
TDI_TARGETS = ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']


def load_auxiliary(data_dir):
    """The eight pIC50 columns from the challenge file, indexed by molecule.

    This is the same file `prepare_data.py` downloads and splits; nothing external is
    introduced. Duplicated molecule names are averaged rather than dropped, so a repeated
    measurement does not silently pick whichever row pandas saw last."""
    path = Path(data_dir) / RAW
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing; run prepare_data.py first')
    frame = pd.read_csv(path)
    missing = [c for c in ['Molecule_Name', *SLOTS] if c not in frame.columns]
    if missing:
        raise KeyError(f'{path} does not carry {missing}')
    return frame.groupby('Molecule_Name')[SLOTS].mean()


class WithAuxiliary(Dataset):
    """Whatever the wrapped dataset returns, plus `aux` [8] for that row."""

    def __init__(self, inner, aux):
        self.inner, self.aux = inner, torch.from_numpy(aux)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return {**self.inner[i], 'aux': self.aux[i]}


def collate_with_auxiliary(items):
    """`collate_atoms` plus the auxiliary block, which it does not know about."""
    batch = collate_atoms(items)
    if 'aux' in items[0]:
        batch['aux'] = torch.stack([item['aux'] for item in items])
    return batch


class TDIAtomJEPAModel(CachedAtomJEPAModel):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        # The head is always eight wide: the slots are the model, and the task's own
        # outputs are read out of them rather than predicted directly.
        super().__init__(cfg, readout, len(SLOTS), data_dir, device)
        self.n_targets = n_targets
        self.structured = bool(cfg.tdi_head)
        if self.structured and n_targets != len(TDI_TARGETS):
            # Guard against this model being pointed at the wrong task: the structured
            # logit only makes sense for the two is_TDI endpoints.
            raise ValueError(f'tdi_head=true expects the {len(TDI_TARGETS)} classification '
                             f'targets, got {n_targets}')
        if not self.structured and n_targets != len(DIRECT):
            raise ValueError(f'tdi_head=false expects the 4 regression targets, '
                             f'got {n_targets}')
        if cfg.aux_pic50 < 0:
            raise ValueError(f'aux_pic50 is a loss weight, got {cfg.aux_pic50}')

        self.register_buffer('slot_mean', torch.zeros(len(SLOTS)))
        self.register_buffer('slot_std', torch.ones(len(SLOTS)))
        # The decision boundary, in the standardized units the head works in. A buffer and
        # not a constant because it is 0.30 divided by a std measured on the training split.
        self.register_buffer('offset', torch.zeros(len(ENZYMES)))
        if self.structured:
            # softplus keeps the slope positive, so a larger shift always means more
            # time-dependent inhibition and the sign of the relation cannot be learned away.
            self.logit_scale = nn.Parameter(torch.full((n_targets,), 2.0))
            self.logit_bias = nn.Parameter(torch.zeros(n_targets))
        self.auxiliary = load_auxiliary(data_dir)
        self.train_molecules = None
        self.target_slots = None          # all three filled in by loader()
        self.logit_slots = None
        self.logit_enzymes = None
        self._stash = None
        self.to(self._device)

    # ---- fitting on the training split only ------------------------------------------

    def fit_preprocessing(self, train):
        super().fit_preprocessing(train)
        self.train_molecules = frozenset(train['Molecule_Name'])
        rows = self.auxiliary.reindex(sorted(self.train_molecules)).to_numpy(np.float64)
        counts = np.isfinite(rows).sum(0)
        if (counts < 2).any():
            thin = [SLOTS[i] for i, c in enumerate(counts) if c < 2]
            raise ValueError(f'too few training values to standardize {thin}')
        if self.structured:
            # One shared scale across all eight: the logit is a difference, and dividing
            # the two terms by different numbers would make that difference meaningless.
            finite = rows[np.isfinite(rows)]
            mean = np.full(len(SLOTS), finite.mean())
            std = np.full(len(SLOTS), max(finite.std(), 1e-6))
        else:
            mean = np.nanmean(rows, axis=0)
            std = np.maximum(np.nanstd(rows, axis=0), 1e-6)
        self.slot_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.slot_std.copy_(torch.as_tensor(std, dtype=torch.float32))
        # 0.30 pIC50 units is the two-fold IC50 shift the training labels agree with for
        # both enzymes; expressed in the head's units it is 0.30 / sigma.
        self.offset.fill_(float(self.cfg.tdi_threshold) / float(std[0] if self.structured
                                                                else 1.0))
        observed = {SLOTS[i]: int(c) for i, c in enumerate(counts)}
        print(f'direction D: {len(self.train_molecules)} training molecules, auxiliary '
              f'coverage {observed}', flush=True)

    # ---- the model interface ---------------------------------------------------------

    def _auxiliary_block(self, frame):
        """[n_rows, 8] standardized auxiliary targets, NaN outside the training split.

        Restricting to `self.train_molecules` is what makes the auxiliary loss unable to
        touch a validation or test label, whatever frame it is handed."""
        if self.train_molecules is None:
            raise RuntimeError('fit_preprocessing must run before loader')
        values = self.auxiliary.reindex(frame['Molecule_Name']).to_numpy(np.float32)
        outside = ~frame['Molecule_Name'].isin(self.train_molecules).to_numpy()
        values[outside] = np.nan
        mean = self.slot_mean.detach().cpu().numpy()
        std = self.slot_std.detach().cpu().numpy()
        return (values - mean) / std

    def loader(self, frame, targets, batch_size, shuffle=False):
        names = list(targets)
        if self.structured:
            unknown = [t for t in names if t not in TDI_LABEL]
            if unknown:
                raise ValueError(f'tdi_head=true cannot read targets {unknown}')
            # For enzyme e the logit needs its direct slot and its TDI slot.
            self.logit_slots = [(SLOTS.index(f'{TDI_LABEL[t]}_pIC50_direct_inhibition'),
                                 SLOTS.index(f'{TDI_LABEL[t]}_pIC50_TDI_condition'))
                                for t in names]
            self.logit_enzymes = [ENZYMES.index(TDI_LABEL[t]) for t in names]
        else:
            unknown = [t for t in names if t not in DIRECT]
            if unknown:
                raise ValueError(f'tdi_head=false cannot read targets {unknown}')
            self.target_slots = [SLOTS.index(t) for t in names]
        y = frame.reindex(columns=names).to_numpy(dtype=np.float32)
        aux = self._auxiliary_block(frame)
        if self.cfg.source == 'pooled':
            data = with_interval(PooledSet(self._pooled(frame), y), frame, names)
            # torch's default collator stacks the extra tensor field on its own.
            return DataLoader(WithAuxiliary(data, aux), batch_size=batch_size,
                              shuffle=shuffle)
        data = AtomSet(self.cache, self.cache.rows(frame), self.layers, self.degrees, y)
        data = WithAuxiliary(with_interval(data, frame, names), aux)
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_with_auxiliary)

    def forward(self, batch):
        # The parent's forward ends in self.readout, so calling it gives the eight slots.
        slots = CachedAtomJEPAModel.forward(self, batch)
        self._stash = (slots, batch.get('aux')) if self.training else None
        if not self.structured:
            return slots[:, self.target_slots]
        direct = slots[:, [pair[0] for pair in self.logit_slots]]
        shifted = slots[:, [pair[1] for pair in self.logit_slots]]
        # Indexed by enzyme, not by target position: the two classification endpoints are
        # CYP2D6 and CYP3A4, i.e. enzymes 2 and 3, so a leading slice would read the wrong
        # pair. The four entries happen to be equal today, which is exactly what would
        # have kept this quiet.
        offset = self.offset[self.logit_enzymes]
        return (nn.functional.softplus(self.logit_scale) * (shifted - direct - offset)
                + self.logit_bias)

    def penalty(self):
        """Auxiliary MSE on the pIC50 slots, over observed TRAINING labels only.

        Delivered through the `penalty()` hook finetune.py already calls, so the shared
        training loop needs no change. `train_loss` in the history therefore stays the data
        term and remains comparable with every other run."""
        if not self.cfg.aux_pic50 or self._stash is None:
            return None
        slots, aux = self._stash
        if aux is None:
            return None
        aux = aux.to(slots.device, slots.dtype)
        if not self.structured:
            # Slots 0..3 are already supervised by the task loss in finetune.py's own
            # standardized space; adding a second term on them would just reweight it.
            slots, aux = slots[:, TDI_SLOTS], aux[:, TDI_SLOTS]
        mask = torch.isfinite(aux)
        if not mask.any():
            return None
        return self.cfg.aux_pic50 * nn.functional.mse_loss(slots[mask], aux[mask])
