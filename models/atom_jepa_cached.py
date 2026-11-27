"""Atom-JEPA with a frozen encoder, reading the cache from scripts/cache_features.py.

The encoder never runs here, so a full training run costs seconds to minutes instead of
hours: 25 s for 100 epochs on the pooled features, ~500-600 s once `source=per_atom` makes
the head read every atom, against ~2400 s of A100 for 20 epochs of real fine-tuning. That
is what makes it affordable to ask, properly and over several seeds, which part of the
pretrained representation actually carries CYP signal:

  layers   which of the 8 blocks (plus 'last', the default of AtomJEPA.forward) to read
  degrees  l=0 only, as models/atom_jepa.py does today, or also the ||l=1|| and ||l=2||
           norms that carry the geometry an equivariant encoder is built to capture
  pool     mean over atoms, as today, or an attention pooling that can concentrate on the
           few atoms driving inhibition instead of diluting them over ~45 atoms

`layers=[last] degrees=[0] pool=mean` reproduces exactly the representation the starter
feeds its readout, so every comparison has an honest reference point.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from models.readouts import build_readout
from models.targets import with_interval

CHUNK = 256  # molecules per pass when accumulating feature statistics


class FeatureCache:
    """The memmapped arrays written by scripts/cache_features.py."""

    def __init__(self, folder):
        folder = Path(folder)
        if not (folder / 'meta.json').exists():
            raise FileNotFoundError(
                f'No feature cache at {folder}. Build it with:\n'
                f'  python scripts/cache_features.py --device mps --batch-size 16')
        self.meta = json.loads((folder / 'meta.json').read_text())
        self.row = {k: i for i, k in enumerate(np.load(folder / 'keys.npy'))}
        self.offsets = np.load(folder / 'offsets.npy')
        self.pooled = np.load(folder / 'pooled.npy', mmap_mode='r')
        self.per_atom = np.load(folder / 'per_atom.npy', mmap_mode='r')

    def rows(self, frame):
        missing = set(frame.key) - self.row.keys()
        if missing:
            raise KeyError(f'{len(missing)} molecules are not in the feature cache')
        return np.array([self.row[k] for k in frame.key])

    def layer_index(self, name):
        names = self.meta['layers']
        if str(name) not in names:
            raise ValueError(f'layer {name!r} not cached; available: {names}')
        return names.index(str(name))


def segment_softmax(scores, index, num_graphs):
    """Softmax of [N, H] scores within each graph, max-shifted for stability."""
    target = index.unsqueeze(-1).expand_as(scores)
    top = scores.new_full((num_graphs, scores.shape[1]), float('-inf'))
    top = top.scatter_reduce(0, target, scores, reduce='amax', include_self=True)
    weight = (scores - top[index]).exp()
    total = torch.zeros_like(top).index_add_(0, index, weight)
    return weight / total[index].clamp_min(1e-12)


class AttentionPool(nn.Module):
    """Gated attention pooling over the atoms of each molecule.

    Follows attention-based multiple-instance learning (Ilse et al., 2018): the molecule is
    a bag of atoms and the head learns which atoms to weigh. The gate (the sigmoid branch)
    lets it suppress atoms outright, which plain tanh attention cannot do."""

    def __init__(self, dim, hidden=128, heads=1, gated=True):
        super().__init__()
        self.heads = heads
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden) if gated else None
        self.score = nn.Linear(hidden, heads)

    def forward(self, x, index, num_graphs):
        hidden = torch.tanh(self.value(x))
        if self.gate is not None:
            hidden = hidden * torch.sigmoid(self.gate(x))
        attention = segment_softmax(self.score(hidden), index, num_graphs)   # [N, heads]
        weighted = x.unsqueeze(1) * attention.unsqueeze(-1)                  # [N, heads, dim]
        pooled = torch.zeros(num_graphs, self.heads, x.shape[1],
                             dtype=x.dtype, device=x.device).index_add_(0, index, weighted)
        return pooled.flatten(1)


class ScalarMix(nn.Module):
    """One learned softmax weight per layer, as in ELMo: a differentiable layer choice."""

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):                      # [..., n_layers, dim] -> [..., dim]
        w = self.weights.softmax(0).view(*([1] * (x.dim() - 2)), -1, 1)
        return self.scale * (x * w).sum(-2)


class PooledSet(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {'features': self.X[i], 'y': self.y[i]}


class AtomSet(Dataset):
    """Per-atom features, sliced from the memmap on demand to keep memory flat."""

    def __init__(self, cache, rows, layers, degrees, y):
        self.cache, self.rows, self.layers, self.degrees = cache, rows, layers, degrees
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        block = self.cache.per_atom[self.cache.offsets[r]:self.cache.offsets[r + 1]]
        block = np.asarray(block[:, self.layers][:, :, self.degrees], dtype=np.float32)
        return {'features': torch.from_numpy(block).flatten(2), 'y': self.y[i]}


def collate_atoms(items):
    counts = [len(item['features']) for item in items]
    index = torch.repeat_interleave(torch.arange(len(items)), torch.tensor(counts))
    batch = {'features': torch.cat([item['features'] for item in items]),
             'node_graph_index': index, 'num_graphs': len(items),
             'y': torch.stack([item['y'] for item in items])}
    if 'bounds' in items[0]:   # present for regression only, see models/targets.py
        batch['bounds'] = torch.stack([item['bounds'] for item in items])
    return batch


class CachedAtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self.cache = FeatureCache(Path(data_dir) / 'features' / cfg.cache)
        self._device = torch.device(device)

        names = list(cfg.layers) if cfg.layers != 'all' else self.cache.meta['layers']
        self.layers = [self.cache.layer_index(n) for n in names]
        self.degrees = list(cfg.degrees)
        if not self.layers or not self.degrees:
            raise ValueError('select at least one layer and one degree')
        if not set(self.degrees) <= {0, 1, 2}:
            raise ValueError('degrees are 0 (l=0), 1 (||l=1||) and 2 (||l=2||)')
        if cfg.source not in ('pooled', 'per_atom'):
            raise ValueError(f'unknown source {cfg.source!r}')
        if cfg.source == 'pooled' and cfg.pool != 'mean':
            raise ValueError("pool != 'mean' needs source=per_atom")

        channels = self.cache.meta['channels']
        width = len(self.degrees) * channels          # features per layer
        self.register_buffer('feature_mean', torch.zeros(len(self.layers), width))
        self.register_buffer('feature_std', torch.ones(len(self.layers), width))

        self.mix = ScalarMix(len(self.layers)) if cfg.layer_mix == 'scalar_mix' else None
        if cfg.layer_mix not in ('concat', 'scalar_mix'):
            raise ValueError(f'unknown layer_mix {cfg.layer_mix!r}')
        dim = width if self.mix is not None else width * len(self.layers)

        self.pool = None
        if cfg.pool in ('attention', 'gated_attention'):
            self.pool = AttentionPool(dim, hidden=cfg.pool_hidden, heads=cfg.pool_heads,
                                      gated=cfg.pool == 'gated_attention')
            dim *= cfg.pool_heads
        elif cfg.pool not in ('mean', 'sum'):
            raise ValueError(f'unknown pool {cfg.pool!r}')

        self.readout = build_readout(readout, dim, n_targets)
        self.to(self._device)

    @property
    def device(self):
        return self._device

    # ---- features -------------------------------------------------------------------

    def _pooled(self, frame):
        """[n_molecules, n_layers, width] straight from the pooled cache."""
        rows = self.cache.rows(frame)
        X = np.asarray(self.cache.pooled[rows][:, self.layers][:, :, self.degrees],
                       dtype=np.float32)
        return torch.from_numpy(X).flatten(2)

    def fit_preprocessing(self, train):
        """Per-feature mean and std from the TRAINING split only."""
        if self.cfg.scaling == 'none':
            return
        if self.cfg.scaling != 'standard':
            raise ValueError(f'unknown scaling {self.cfg.scaling!r}')
        if self.cfg.source == 'pooled':
            X = self._pooled(train)
            mean, std = X.mean(0), X.std(0)
        else:
            rows = self.cache.rows(train)
            total = torch.zeros_like(self.feature_mean, dtype=torch.float64)
            square = torch.zeros_like(total)
            n = 0
            for start in range(0, len(rows), CHUNK):
                for r in rows[start:start + CHUNK]:
                    block = self.cache.per_atom[self.cache.offsets[r]:self.cache.offsets[r + 1]]
                    x = torch.from_numpy(
                        np.asarray(block[:, self.layers][:, :, self.degrees],
                                   dtype=np.float64)).flatten(2)
                    total += x.sum(0)
                    square += x.pow(2).sum(0)
                    n += len(x)
            mean = (total / n).float()
            std = (square / n - (total / n) ** 2).clamp_min(0).sqrt().float()
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.clamp_min(1e-6).to(self.feature_std))

    def _embed(self, features):
        """[*, n_layers, width] -> [*, dim], standardized and mixed across layers."""
        x = (features - self.feature_mean) / self.feature_std
        return self.mix(x) if self.mix is not None else x.flatten(-2)

    # ---- the model interface --------------------------------------------------------

    def loader(self, frame, targets, batch_size, shuffle=False):
        y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        if self.cfg.source == 'pooled':
            data = with_interval(PooledSet(self._pooled(frame), y), frame, targets)
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        data = AtomSet(self.cache, self.cache.rows(frame), self.layers, self.degrees, y)
        return DataLoader(with_interval(data, frame, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate_atoms)

    def param_groups(self):
        return [{'params': list(self.parameters()), 'lr': self.cfg.readout_lr}]

    def forward(self, batch):
        x = self._embed(batch['features'].to(self.device))
        if self.cfg.source == 'per_atom':
            index = batch['node_graph_index'].to(self.device)
            graphs = batch['num_graphs']
            if self.pool is not None:
                x = self.pool(x, index, graphs)
            else:
                pooled = torch.zeros(graphs, x.shape[1], dtype=x.dtype,
                                     device=x.device).index_add_(0, index, x)
                if self.cfg.pool == 'mean':
                    counts = torch.zeros(graphs, 1, dtype=x.dtype, device=x.device)
                    counts.index_add_(0, index, torch.ones_like(x[:, :1]))
                    pooled = pooled / counts.clamp_min(1)
                x = pooled
        return self.readout(x)
