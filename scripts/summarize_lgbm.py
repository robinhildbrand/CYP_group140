#!/usr/bin/env python3
"""Export one row per LightGBM validation run, keeping *all* Hydra settings.

Unlike scripts/aggregate_results.py, this script does not group runs by model.name:
the same name is used for many different LightGBM hyperparameters/feature ablations.

Examples (from the group repository root):
    python scripts/summarize_lgbm.py --runs ../dtu-cyp-project/runs
    python scripts/summarize_lgbm.py --runs ../dtu-cyp-project/runs --csv /tmp/lgbm-runs.csv
No test labels or test metrics are consulted.
"""
import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = [
    'task', 'run', 'seed', 'features', 'num_leaves', 'learning_rate',
    'min_child_samples', 'reg_lambda', 'class_weight', 'n_estimators',
    'early_stopping_rounds', 'metric', 'validation_score',
]


def collect(root):
    rows = []
    for config_path in sorted(root.glob('*/*/*/config.json')):
        run = config_path.parent
        metrics_path = run / 'validation_metrics.json'
        if not metrics_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        model = config.get('model', {})
        if model.get('type') != 'lightgbm':
            continue
        task = config.get('task')
        primary = {'regression': 'macro_ST-RAE', 'classification': 'macro_MCC'}.get(task)
        if primary is None:
            continue
        score = json.loads(metrics_path.read_text()).get(primary)
        if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        p = model.get('params', {})
        rows.append({
            'task': task,
            'run': str(run),
            'seed': config.get('seed'),
            'features': '+'.join(model.get('featurizers', [])),
            'num_leaves': p.get('num_leaves'),
            'learning_rate': p.get('learning_rate'),
            'min_child_samples': p.get('min_child_samples'),
            'reg_lambda': p.get('reg_lambda'),
            'class_weight': p.get('class_weight', 'none'),
            'n_estimators': p.get('n_estimators'),
            'early_stopping_rounds': model.get('early_stopping_rounds'),
            'metric': primary,
            'validation_score': score,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=Path('runs'), help='Directory containing task/model/timestamp runs')
    parser.add_argument('--csv', type=Path, help='Optional destination for the per-run CSV')
    args = parser.parse_args()
    rows = collect(args.runs)
    if not rows:
        parser.exit(1, f'No LightGBM validation metrics found in {args.runs}\n')
    for task in ('regression', 'classification'):
        matching = sorted([r for r in rows if r['task'] == task],
                          key=lambda r: r['validation_score'], reverse=(task == 'classification'))
        print(f'\n{task}: {len(matching)} runs (validation only)')
        for r in matching:
            print(f"  {r['validation_score']:.6f}  features={r['features']} "
                  f"leaves={r['num_leaves']} lr={r['learning_rate']} "
                  f"min_child={r['min_child_samples']} lambda={r['reg_lambda']} "
                  f"weights={r['class_weight']} seed={r['seed']} "
                  f"run={Path(r['run']).name}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nWrote {len(rows)} experiments to {args.csv}')


if __name__ == '__main__':
    main()
