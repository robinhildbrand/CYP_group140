#!/usr/bin/env python3
"""Validation-only LightGBM threshold sensitivity study; NOT the official test scorer.

The DTU project specifies the supplied evaluator's fixed 0.5 threshold for the
final comparison. This helper quantifies sensitivity to alternative thresholds
on validation and does not change finetune.py or the official scores.

Example (from the group repository root):
    python scripts/analyze_lgbm_thresholds.py \\
      --run ../dtu-cyp-project/runs/classification/morgan_rdkit_lgbm/2026-11-14_15-53-23 \\
      --labels ../dtu-cyp-project/data/classification/val.csv \\
      --selected CYP2D6_is_TDI=0.36 CYP3A4_is_TDI=0.17

Never use test labels or test predictions with this script.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

TARGETS = ('CYP2D6_is_TDI', 'CYP3A4_is_TDI')


def parse_selected(values):
    result = {}
    for item in values:
        name, separator, threshold = item.partition('=')
        if separator != '=' or name not in TARGETS:
            raise ValueError(f'Expected one of {TARGETS} as TARGET=THRESHOLD, got {item!r}')
        value = float(threshold)
        if not 0 <= value <= 1:
            raise ValueError(f'Invalid threshold {value} for {name}')
        result[name] = value
    return result


def load_validation_predictions(labels_path, prediction_path):
    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(prediction_path)
    # finetune.py filters out rows without any target label before prediction;
    # it then exports predictions in that filtered validation-row order.
    expected = labels.loc[labels[list(TARGETS)].notna().any(axis=1)].reset_index(drop=True)
    if len(expected) != len(predictions):
        raise ValueError('Validation row counts differ. Check the run and data split.')
    for col in ('Molecule_Name', 'SMILES'):
        left = expected[col].fillna('').astype(str).reset_index(drop=True)
        right = predictions[col].fillna('').astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f'Validation {col} order/content differs from predictions.')
    return expected, predictions


def analyze(expected, probabilities, chosen):
    rows = []
    for target in TARGETS:
        mask = expected[target].notna().to_numpy()
        y = expected.loc[mask, target].to_numpy(dtype=int)
        p = probabilities.loc[mask, target].to_numpy(dtype=float)
        if not np.isfinite(p).all() or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f'Nonfinite probabilities or nonbinary labels: {target}')
        default = float(matthews_corrcoef(y, p >= 0.5))
        scores = [(round(float(t), 2), float(matthews_corrcoef(y, p >= t)))
                  for t in np.round(np.arange(0.05, 0.951, 0.01), 2)]
        # When two thresholds yield the same maximum, select the lower one.
        best_threshold, best_mcc = max(scores, key=lambda pair: (pair[1], -pair[0]))
        threshold = chosen.get(target)
        selected_mcc = float(matthews_corrcoef(y, p >= threshold)) if threshold is not None else None
        rows.append({'target': target, 'n': len(y), 'default_threshold': 0.5,
                     'default_MCC': default, 'grid_best_threshold': best_threshold,
                     'grid_best_MCC': best_mcc, 'selected_threshold': threshold,
                     'selected_MCC': selected_mcc})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Classification LightGBM run directory')
    parser.add_argument('--labels', type=Path, required=True, help='Classification validation labels CSV')
    parser.add_argument('--selected', nargs='*', default=[], metavar='TARGET=THRESHOLD',
                        help='Thresholds selected on validation, to report alongside the sweep')
    parser.add_argument('--csv', type=Path, help='Optional per-target output CSV')
    args = parser.parse_args()
    if 'test' in args.labels.name.lower():
        parser.error('This study only accepts validation labels, never test labels.')
    chosen = parse_selected(args.selected)
    labels, probabilities = load_validation_predictions(args.labels,
                                                        args.run / 'validation_predictions.csv')
    rows = analyze(labels, probabilities, chosen)
    for row in rows:
        print(f"{row['target']}: n={row['n']} MCC@0.50={row['default_MCC']:.4f} "
              f"grid_best={row['grid_best_threshold']:.2f} "
              f"MCC={row['grid_best_MCC']:.4f}")
        if row['selected_threshold'] is not None:
            print(f"  selected={row['selected_threshold']:.2f}, "
                  f"MCC={row['selected_MCC']:.4f}")
    mean = lambda field: float(np.mean([r[field] for r in rows]))
    print(f"Macro-MCC @0.50: {mean('default_MCC'):.6f}")
    print(f"Macro-MCC with grid-selected thresholds on SAME validation: {mean('grid_best_MCC'):.6f}")
    if all(row['selected_MCC'] is not None for row in rows):
        print(f"Macro-MCC with supplied thresholds: {mean('selected_MCC'):.6f}")
    print('CAUTION: threshold search and evaluation use the same validation split; '
          'the official test metric remains fixed at 0.5.')
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {args.csv}')


if __name__ == '__main__':
    main()
