"""Per-row credible intervals, carried alongside the labels.

The regression task is scored with ST-RAE, which counts only the part of a prediction's
error that falls OUTSIDE the credible interval of that particular label. Training that
objective therefore needs `<target>_conf_low` and `<target>_conf_high` per row and per
endpoint, not just the label — so a dataset has to hand them to the loss.

`with_interval` wraps any dataset and adds a `bounds` field to each item. Both collators in
this project stack unknown tensor fields along a new batch dimension (torch's default one,
and atom-jepa's GraphCollator), so nothing else changes; `models/atom_jepa_cached.py` adds
the one line its hand-written collator needs. A frame without the interval columns, i.e.
the classification task, is passed through unwrapped and the loss falls back to the labels.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def has_intervals(frame, targets):
    return all(f'{t}_conf_{side}' in frame.columns for t in targets for side in ('low', 'high'))


def intervals(frame, targets):
    """[n_rows, n_targets, 2] of (conf_low, conf_high).

    Non-finite wherever the label itself is missing, which is exactly where the loss masks
    the entry out, so the gaps never reach the arithmetic."""
    return np.stack([frame.reindex(columns=[f'{t}_conf_low', f'{t}_conf_high'])
                     .to_numpy(dtype=np.float32) for t in targets], axis=1)


class WithInterval(Dataset):
    """Whatever the wrapped dataset returns, plus `bounds` [n_targets, 2] for that row."""

    def __init__(self, inner, bounds):
        self.inner, self.bounds = inner, torch.from_numpy(bounds)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return {**self.inner[i], 'bounds': self.bounds[i]}


def with_interval(inner, frame, targets):
    """`inner` with the intervals attached, or unchanged if the frame does not carry them."""
    if not has_intervals(frame, targets):
        return inner
    return WithInterval(inner, intervals(frame, targets))
