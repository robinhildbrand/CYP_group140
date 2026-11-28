#!/usr/bin/env python3
"""Collect every run under runs/ into one comparison table, averaged over seeds.

Runs are grouped by model name with the trailing _s<seed> stripped, so the three seeds of
one configuration become one row with a mean and a standard deviation. The validation set
is small (~720 molecules with sparse labels), so a difference smaller than the spread
across seeds is not a result.

    python scripts/aggregate_results.py                     # validation, both tasks
    python scripts/aggregate_results.py --prefix A1_        # one sweep stage
    python scripts/aggregate_results.py --split test        # only for selected models
    python scripts/aggregate_results.py --csv results.csv
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PRIMARY = {'regression': 'macro_ST-RAE', 'classification': 'macro_MCC'}
LOWER_IS_BETTER = {'regression': True, 'classification': False}
PER_TARGET = {'regression': 'ST-RAE', 'classification': 'MCC'}


def read_runs(root, split):
    rows = []
    for config_path in sorted(root.glob('*/*/*/config.json')):
        directory = config_path.parent
        metrics_path = directory / f'{split}_metrics.json'
        if not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        task = config['task']
        name = config['model']['name']
        row = {'task': task, 'name': re.sub(r'_s\d+$', '', name), 'seed': config['seed'],
               'score': metrics.get(PRIMARY[task]), 'run': str(directory)}
        for target, values in metrics.items():
            if isinstance(values, dict):
                row[target.split('_')[0]] = values.get(PER_TARGET[task])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # A configuration can be run twice on the same seed — a job resubmitted to a faster
    # queue, a crash, a changed default. Counting both would weight that seed double and
    # report a spread narrower than it is, so the newest run wins. The glob is sorted and
    # the run directory is a timestamp, hence keep='last'.
    before = len(frame)
    frame = frame.sort_values('run').drop_duplicates(['task', 'name', 'seed'], keep='last')
    if before > len(frame):
        print(f'{before - len(frame)} superseded run(s) ignored: same configuration and '
              f'seed run more than once, keeping the most recent')
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, default=Path('runs'))
    p.add_argument('--split', default='validation', choices=['validation', 'test'])
    p.add_argument('--prefix', default='', help='Only names starting with this')
    p.add_argument('--csv', type=Path, help='Also write the per-run table here')
    p.add_argument('--top', type=int, help='Show only the best N configurations')
    args = p.parse_args()

    frame = read_runs(args.runs, args.split)
    if frame.empty:
        print(f'No {args.split} metrics under {args.runs}')
        return
    frame = frame[frame.name.str.startswith(args.prefix)]
    if frame.empty:
        print(f'No run matches prefix {args.prefix!r}')
        return
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f'Per-run table -> {args.csv}')

    for task, part in frame.groupby('task'):
        metric = PRIMARY[task]
        better = 'lower is better' if LOWER_IS_BETTER[task] else 'higher is better'
        enzymes = [c for c in part.columns if c.startswith('CYP') and part[c].notna().any()]
        summary = part.groupby('name').agg(
            n=('seed', 'nunique'), mean=('score', 'mean'), std=('score', 'std'),
            **{e: (e, 'mean') for e in enzymes})
        summary = summary.sort_values('mean', ascending=LOWER_IS_BETTER[task])
        if args.top:
            summary = summary.head(args.top)
        print(f'\n{task}  —  {metric} on {args.split} ({better})')
        header = f'{"configuration":28s} {"seeds":>5s} {"mean":>8s} {"std":>7s}'
        print(header + ''.join(f' {e:>9s}' for e in enzymes))
        print('-' * (len(header) + 10 * len(enzymes)))
        for name, row in summary.iterrows():
            std = '' if np.isnan(row['std']) else f'{row["std"]:7.4f}'
            print(f'{name:28s} {int(row["n"]):5d} {row["mean"]:8.4f} {std:>7s}'
                  + ''.join(f' {row[e]:9.4f}' if pd.notna(row[e]) else f' {"":>9s}'
                            for e in enzymes))


if __name__ == '__main__':
    main()
