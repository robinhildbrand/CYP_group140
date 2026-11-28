"""Readout heads: map molecule embeddings [batch, in_dim] to predictions [batch, n_targets].

To add a head, write a class taking (in_dim, n_targets, **options), register it in
READOUTS and add conf/readout/<name>.yaml. Use it with e.g. readout=<name>."""
from torch import nn


class LinearReadout(nn.Module):
    def __init__(self, in_dim, n_targets):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_targets)

    def forward(self, x):
        return self.linear(x)


class MLPReadout(nn.Module):
    """num_layers Linear layers (including the output layer) with ReLU and dropout."""

    def __init__(self, in_dim, n_targets, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], n_targets))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPReadout(nn.Module):
    """Pre-norm residual MLP, an alternative to the plain stacked MLP above.

    Each block is LayerNorm -> Linear -> GELU -> Dropout -> Linear added back to its input,
    so depth no longer makes the head harder to optimize and the normalization keeps the
    very differently scaled Atom-JEPA feature groups (l=0 values and l>0 norms differ by
    roughly 3x in magnitude) on comparable footing."""

    def __init__(self, in_dim, n_targets, hidden_dim=512, num_blocks=2, dropout=0.1,
                 expansion=2):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden_dim)
        inner = hidden_dim * expansion
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, inner), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(inner, hidden_dim))
            for _ in range(num_blocks))
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, n_targets)

    def forward(self, x):
        x = self.project(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(self.norm(x))


READOUTS = {'linear': LinearReadout, 'mlp': MLPReadout, 'resmlp': ResidualMLPReadout}


def build_readout(cfg, in_dim, n_targets):
    options = {k: v for k, v in cfg.items() if k != 'type'}
    return READOUTS[cfg.type](in_dim, n_targets, **options)
