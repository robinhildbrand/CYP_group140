"""Atom-JEPA with a configurable adaptation depth — Direction B.

`models/atom_jepa.py` stays the reference: the whole encoder frozen, or the whole encoder
moving at a single learning rate. Both remain reachable here, and four orthogonal knobs
are added, each answering a different question about how to adapt a pretrained encoder to
3.4k labelled molecules:

| knob              | question                                          | off value |
|-------------------|---------------------------------------------------|-----------|
| `unfreeze_blocks` | how much of the encoder may move at all           | `all`     |
| `llrd`            | how fast each depth moves relative to the one above | `1.0`   |
| `probe_epochs`    | when the encoder starts moving (LP-FT)            | `0`       |
| `l2sp`            | how far it may drift from the pretrained weights  | `0.0`     |

Parameters are grouped by depth, from the atom/edge input path (0) through the eight
transformer blocks (1-8) to the output norm (9). Early layers of a pretrained encoder hold
the generic chemistry and late ones the part specific to the pretraining objective, which
is the ordering both LLRD and last-k unfreezing act on.

With every knob at its off value this model trains exactly like `models/atom_jepa.py`
with `freeze_encoder=false`; `unfreeze_blocks=0` reproduces `freeze_encoder=true`.
"""
import re

import torch
from torch import nn
from torch.utils.data import DataLoader
from atom_jepa import AtomJEPA
from atom_jepa.checkpoint import load_pretrained_encoder
from models.atom_jepa import Conformers
from models.atom_jepa_cached import AttentionPool, ScalarMix
from models.readouts import build_readout
from models.targets import with_interval


def depth_of(name, num_blocks):
    """Depth of an encoder parameter, bottom to top.

    0 is the atom-type and edge-degree input path, 1..num_blocks the transformer blocks,
    num_blocks + 1 the output norm. Unknown names raise instead of being silently filed
    at depth 0, so a change in the atom-jepa package shows up as an error here rather than
    as a quietly wrong learning rate."""
    block = re.match(r'body\.blocks\.(\d+)\.', name)
    if block:
        index = int(block.group(1))
        if index >= num_blocks:
            raise KeyError(f'Encoder parameter {name!r} is past block {num_blocks - 1}')
        return index + 1
    if name.startswith('body.norm.'):
        return num_blocks + 1
    if name.startswith(('body.sphere_embedding.', 'body.edge_degree_embedding.')):
        return 0
    raise KeyError(f'Unexpected encoder parameter {name!r}; update the depth grouping')


def build_encoder(cfg, device):
    """The released encoder, or the same architecture at fresh random init.

    `pretrained=false` is the control that separates "the pretraining helped" from "this
    equivariant architecture helped": the checkpoint still provides the architecture, so
    the two runs differ in nothing but the initial weights."""
    model = AtomJEPA.from_pretrained(cfg.checkpoint, device=device, precision=cfg.precision,
                                     cuequivariance=False)
    if not cfg.pretrained:
        fresh, _, _ = load_pretrained_encoder(cfg.checkpoint, device, load_weights=False)
        model.encoder.load_state_dict(fresh.state_dict())
    return model


class FineTunedAtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'conformers'
        self.encoder = build_encoder(cfg, device)
        # AtomJEPA.encoder is the EquiformerV3Encoder; its parameters are named body.*.
        body = self.encoder.encoder
        self.num_blocks = len(body.body.blocks)
        self.top = self.num_blocks + 1
        if cfg.llrd <= 0 or cfg.llrd > 1:
            raise ValueError(f'llrd is a decay factor in (0, 1], got {cfg.llrd}')
        if cfg.probe_epochs < 0 or cfg.l2sp < 0:
            raise ValueError('probe_epochs and l2sp cannot be negative')

        # Which depths may move. 'all' is the whole encoder, input path included. An integer
        # k is the top k transformer blocks plus the output norm sitting above them, so
        # k=0 leaves the encoder completely frozen: the linear-probe end of the axis.
        if cfg.unfreeze_blocks == 'all':
            floor = 0
        elif not 0 <= int(cfg.unfreeze_blocks) <= self.num_blocks:
            raise ValueError(f'unfreeze_blocks must be 0..{self.num_blocks} or "all", '
                             f'got {cfg.unfreeze_blocks!r}')
        elif int(cfg.unfreeze_blocks) == 0:
            floor = self.top + 1
        else:
            floor = self.top - int(cfg.unfreeze_blocks)
        self.trainable = {}
        for name, p in body.named_parameters():
            depth = depth_of(name, self.num_blocks)
            if depth >= floor:
                self.trainable.setdefault(depth, []).append(p)
        # L2-SP anchors the moving weights to where pretraining left them. Kept outside
        # the module tree on purpose: registering 39M buffers would double every best.pt.
        self.anchors = ([(p, p.detach().clone()) for params in self.trainable.values()
                         for p in params] if cfg.l2sp else [])
        # During a probe phase the encoder is off, not merely on a zero learning rate:
        # requires_grad=False is what actually saves the backward pass through it.
        self.awake = not cfg.probe_epochs
        self.encoder.requires_grad_(False)
        for params in self.trainable.values():
            for p in params:
                p.requires_grad_(self.awake)
        self.encoder.set_grad_checkpointing(bool(self.trainable) and self.awake)
        # How the atoms of a molecule are reduced to one vector. The mean is what
        # AtomJEPA.forward does and stays the default, so every knob at its off value still
        # reproduces models/atom_jepa.py exactly. Attention pooling is the same module the
        # frozen-feature experiments used, where it beat the mean by 0.038 ST-RAE and was
        # the only thing that moved CYP2D6 — but it had never been tried on an encoder that
        # is itself being updated, which is what this option is for.
        width = self.encoder.embedding_dim
        if cfg.pool == 'mean':
            self.pool = None
        elif cfg.pool in ('attention', 'gated_attention'):
            if cfg.pool_heads < 1:
                raise ValueError(f'pool_heads must be positive, got {cfg.pool_heads}')
            self.pool = AttentionPool(width, heads=cfg.pool_heads,
                                      gated=cfg.pool == 'gated_attention').to(
                                          self.encoder.device)
            width *= cfg.pool_heads
        else:
            raise ValueError(f'pool must be mean, attention or gated_attention, '
                             f'got {cfg.pool!r}')

        # Which depth the head reads. On frozen features the last block was a poor choice —
        # 9th of 11 on classification — and a learned softmax over all of them was the best.
        # That argument is weaker here, because an encoder being updated can reshape its own
        # top block, so this is a hypothesis to test rather than a port of a known win.
        # `layers='all'` returns the 8 transformer blocks; the 'last' slot of the cached
        # experiments is a 9th reading taken after the final norm and is not included.
        if cfg.layer_mix == 'none':
            self.mix = None
        elif cfg.layer_mix == 'scalar_mix':
            if self.pool is None:
                raise ValueError('layer_mix needs an attention pool: mixing happens per '
                                 'atom, and the encoder\'s own mean pooling runs before the '
                                 'head can see the layer axis')
            self.mix = ScalarMix(self.encoder.num_layers).to(self.encoder.device)
        else:
            raise ValueError(f'layer_mix must be none or scalar_mix, got {cfg.layer_mix!r}')
        self.readout = build_readout(readout, width, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

    def train(self, mode=True):
        """The encoder is in train mode only while it is actually being updated.

        Its drop-path is a pretraining regulariser; leaving it on for weights that cannot
        move only injects noise into the features the head is trying to read."""
        super().train(mode)
        if not (self.trainable and self.awake):
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        data = with_interval(Conformers(frame, self.folder, targets), frame, targets)
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.encoder.collate, num_workers=0)

    def param_groups(self):
        """One group per encoder depth, so LLRD is just a per-group learning rate.

        `lr_scale` is the hook the training loop multiplies into the rate every step; it is
        how LP-FT switches the encoder on without rebuilding the optimiser."""
        # The pooling head is learned from scratch like the readout, so it belongs on the
        # readout's rate, not on the encoder's — and it must stay in the group named
        # 'readout', which set_epoch() uses to tell the head apart from the encoder.
        head = list(self.readout.parameters())
        for module in (self.pool, self.mix):
            if module is not None:
                head += list(module.parameters())
        groups = [{'name': 'readout', 'params': head, 'lr': self.cfg.readout_lr}]
        for depth, params in sorted(self.trainable.items()):
            group = {'name': f'encoder.{depth}', 'params': params,
                     'lr': self.cfg.encoder_lr * self.cfg.llrd ** (self.top - depth),
                     'lr_scale': float(self.awake)}
            if self.cfg.l2sp:
                # L2-SP replaces weight decay for these weights; applying both would pull
                # them towards the pretrained values and towards zero at the same time.
                group['weight_decay'] = 0.0
            groups.append(group)
        return groups

    def set_epoch(self, epoch, optimizer):
        """LP-FT (Kumar et al. 2022): the head trains alone for `probe_epochs` epochs, then
        the encoder joins it. A head that still outputs noise sends large, uninformative
        gradients into the encoder and distorts exactly the pretrained features it should
        be exploiting; probing first removes that transient."""
        awake = epoch > self.cfg.probe_epochs
        if awake == self.awake:
            return
        self.awake = awake
        for params in self.trainable.values():
            for p in params:
                p.requires_grad_(awake)
        for group in optimizer.param_groups:
            if group['name'] != 'readout':
                group['lr_scale'] = float(awake)
        self.encoder.set_grad_checkpointing(bool(self.trainable) and awake)
        self.train(self.training)
        print(f'epoch {epoch}: encoder {"unfrozen" if awake else "frozen"}', flush=True)

    def penalty(self):
        """L2-SP (Li et al. 2018), or None when it is switched off or nothing moves yet."""
        if not self.anchors or not self.awake:
            return None
        total = sum((p - anchor).pow(2).sum() for p, anchor in self.anchors)
        return self.cfg.l2sp * total

    def forward(self, batch):
        if self.pool is None:
            # Differentiable; do NOT use encoder.embed(), which detaches.
            return self.readout(self.encoder(batch))
        if self.mix is None:
            atoms = self.encoder(batch, per_atom=True)       # [num_atoms, embedding_dim]
        else:
            atoms = self.encoder(batch, layers='all', per_atom=True)   # [num_atoms, 8, dim]
            atoms = self.mix(atoms.to(self.mix.scale.dtype))
        # The collator builds this on the CPU; the encoder's output is on the GPU, and
        # scatter_reduce refuses a mixed pair. A CPU-only test cannot catch this, which is
        # what the single GPU canary job is for -- it failed here in 26 seconds.
        index = batch['node_graph_index'].to(atoms.device)
        # The pooling head is fp32 while the encoder may be running in bf16, so the
        # features have to meet it rather than the other way round: casting the head down
        # would quantise the attention logits, and a softmax over ~45 atoms in bf16 has
        # about three significant digits to work with.
        atoms = atoms.to(self.pool.value.weight.dtype)
        return self.readout(self.pool(atoms, index, batch['y'].shape[0]))
