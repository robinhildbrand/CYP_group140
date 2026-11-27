#!/usr/bin/env python3
"""Run a Direction-A sweep over frozen Atom-JEPA features, several runs at a time.

Every run goes through finetune.py, so the splits, the masked loss, the metrics and the
best-epoch selection are identical to every other model in the project. Runs are forced to
`eval_test=false`: a sweep selects on validation and never reads the test labels.

Quote any override holding brackets, or zsh will try to expand it as a glob.

    python scripts/sweep.py layers   --jobs 5
    python scripts/sweep.py degrees  --jobs 5 --base 'model.layers=[4]'
    python scripts/sweep.py pool     --jobs 4 --base 'model.layers=[4]' 'model.degrees=[0,1,2]'
    python scripts/sweep.py head     --jobs 5 --seeds 1

Then read the results with scripts/aggregate_results.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from pathlib import Path
import random
import subprocess
import sys
import time

TASKS = ['regression', 'classification']


def layers_stage():
    """Which encoder block carries the CYP signal? 'last' is what the starter uses."""
    for layer in [1, 2, 3, 4, 5, 6, 7, 8, 'last']:
        yield f'A1_layer{layer}', {'model.layers': f'[{layer}]'}
    yield 'A1_all_concat', {'model.layers': 'all'}
    yield 'A1_all_scalarmix', {'model.layers': 'all', 'model.layer_mix': 'scalar_mix'}


def degrees_stage():
    """Does the geometry in the l>0 channels help, or are the l=0 scalars enough?"""
    for name, degrees in [('l0', '[0]'), ('l0_l1', '[0,1]'), ('l0_l1_l2', '[0,1,2]'),
                          ('l1_l2', '[1,2]')]:
        yield f'A3_{name}', {'model.degrees': degrees}


def pool_stage():
    """Mean pooling dilutes a pharmacophore over ~45 atoms; can a head learn where to look?"""
    yield 'A4_mean', {'model.source': 'pooled', 'model.pool': 'mean'}
    yield 'A4_atom_mean', {'model.source': 'per_atom', 'model.pool': 'mean'}
    yield 'A4_atom_sum', {'model.source': 'per_atom', 'model.pool': 'sum'}
    for heads in [1, 4]:
        for kind in ['attention', 'gated_attention']:
            yield (f'A4_{kind}_h{heads}',
                   {'model.source': 'per_atom', 'model.pool': kind, 'model.pool_heads': heads})


def loss_stage():
    """Direction C: neither default objective matches the metric its task is scored on.

    A stage entry may name the tasks it applies to; a regression loss on the classification
    task is a configuration error, not an experiment."""
    yield 'C1_mse', {}, ['regression']
    yield 'C1_huber', {'train.loss': 'huber'}, ['regression']
    yield 'C1_l1', {'train.loss': 'l1'}, ['regression']
    yield 'C1_softthr', {'train.loss': 'soft_threshold'}, ['regression']
    for aux in [0.03, 0.1, 0.3]:
        yield (f'C1_softthr_aux{aux}', {'train.loss': 'soft_threshold',
                                        'train.loss_aux': aux}, ['regression'])
    yield 'C2_bce', {}, ['classification']
    yield 'C2_posw_auto', {'train.pos_weight': 'auto'}, ['classification']
    for weight in [2.0, 3.0, 6.0]:
        yield f'C2_posw{weight}', {'train.pos_weight': weight}, ['classification']


def combine_stage():
    """Do the Direction-A readout and the Direction-C objective add up, or overlap?

    Measured separately against the same mean-pooling baseline, attention pooling took
    regression from 0.7688 to 0.7304 and the soft-threshold loss to 0.7299 — nearly the
    same gain, and both of them moved CYP1A2. That is exactly the shape of two levers that
    might be reading the same signal, so the cross has to be run rather than assumed.
    Each task keeps its own control first, so the table reads as a 2x2 and not as one
    number: AC_attn4_mse repeats A4_attention_h4, which also makes it a reproduction check.
    """
    attention = {'model.source': 'per_atom', 'model.pool': 'attention', 'model.pool_heads': 4}
    yield 'AC_attn4_mse', attention, ['regression']
    yield 'AC_attn4_softthr', {**attention, 'train.loss': 'soft_threshold'}, ['regression']
    yield ('AC_attn4_softthr_aux0.03', {**attention, 'train.loss': 'soft_threshold',
                                        'train.loss_aux': 0.03}, ['regression'])
    yield 'AC_attn4_bce', attention, ['classification']
    yield 'AC_attn4_posw3', {**attention, 'train.pos_weight': 3.0}, ['classification']
    yield 'AC_attn4_posw_auto', {**attention, 'train.pos_weight': 'auto'}, ['classification']


def head_stage(rng, n=20):
    """Random search over the head, free now that the encoder is cached."""
    for i in range(n):
        kind = rng.choice(['mlp', 'resmlp'])
        options = {'readout': kind,
                   'readout.hidden_dim': rng.choice([256, 512, 1024]),
                   'readout.dropout': rng.choice([0.0, 0.1, 0.2, 0.3]),
                   'model.readout_lr': rng.choice([3e-4, 1e-3, 3e-3]),
                   'train.weight_decay': rng.choice([1e-4, 1e-3, 1e-2, 1e-1]),
                   'train.batch_size': rng.choice([32, 64, 128])}
        options['readout.num_blocks' if kind == 'resmlp' else 'readout.num_layers'] = (
            rng.choice([1, 2, 3]))
        yield f'A5_rand{i:02d}', options


# The two readouts the A5 random search elected, measured on three seeds each. They are
# kept here rather than in a yaml because they are results, not defaults: BEST_HEAD only
# means anything next to the control it beat, and the two tasks disagree about which head
# to use. The full record of the search is in results/atom_jepa/validation_runs.csv.
#   regression       A5_rand07  0.6746 +/- 0.0091  against 0.6935 +/- 0.0065  (2.9 sigma)
#   classification   A5_rand11  0.3729 +/- 0.0064  against 0.3687 +/- 0.0111  (noise)
BEST_HEAD = {
    'regression': {'readout': 'mlp', 'readout.hidden_dim': 256, 'readout.num_layers': 2,
                   'readout.dropout': 0.1, 'model.readout_lr': 3.0e-4,
                   'train.weight_decay': 1.0e-3, 'train.batch_size': 128},
    'classification': {'readout': 'mlp', 'readout.hidden_dim': 512, 'readout.num_layers': 3,
                       'readout.dropout': 0.1, 'model.readout_lr': 3.0e-4,
                       'train.weight_decay': 1.0e-3, 'train.batch_size': 32},
}
# The representation every one of those numbers was measured on: Direction A's winner.
BEST_FEATURES = {'model.source': 'per_atom', 'model.layers': 'all',
                 'model.layer_mix': 'scalar_mix', 'model.degrees': '[0]',
                 'model.pool': 'attention', 'model.pool_heads': 4}


def polish_stage():
    """The combinations the stage-by-stage search never put together.

    Each direction was screened on the readout that was current when it ran, so two cheap
    crossings are still missing: the aligned loss's dead-zone companion on top of the
    winning regression head, and the positive-class weight on top of the winning
    classification head. `train.loss_aux` scored the same as plain soft_threshold on the
    old head but with a third of the spread, and `pos_weight` bought 0.020 MCC on mean
    pooling and nothing on attention pooling -- both are worth one measurement on the head
    that actually won rather than an assumption either way.

    The first row of each task reproduces the winner exactly, so the comparison has its own
    control inside the stage instead of relying on a number from another run.
    """
    reg = {**BEST_FEATURES, **BEST_HEAD['regression'], 'train.loss': 'soft_threshold'}
    cls = {**BEST_FEATURES, **BEST_HEAD['classification']}
    yield 'P0_reg_winner', reg, ['regression']
    yield 'P1_reg_aux003', {**reg, 'train.loss_aux': 0.03}, ['regression']
    yield 'P1_reg_aux01', {**reg, 'train.loss_aux': 0.1}, ['regression']
    yield 'P0_cls_winner', cls, ['classification']
    yield 'P2_cls_posw3', {**cls, 'train.pos_weight': 3.0}, ['classification']
    yield 'P2_cls_poswauto', {**cls, 'train.pos_weight': 'auto'}, ['classification']


def tdi_stage():
    """Direction D: does building the TDI label's own definition into the head pay?

    On training rows alone, `is_TDI` is almost exactly
    `pIC50_TDI_condition - pIC50_direct_inhibition > 0.30` (AUC 0.962 on CYP2D6, 0.855 on
    CYP3A4, and the threshold comes out at 0.30 for both independently). D1 builds that
    subtraction and that comparison into the logit instead of asking a 2-wide head to
    rediscover them. D2 asks a narrower question on the regression task: does predicting
    the pre-incubation assay as a side task help the direct one?

    `aux_pic50=0` is the ablation that keeps a reader from confusing the two things this
    direction changes at once -- the shape of the head, and the extra real-valued
    supervision. On regression it doubles as the control for what an eight-wide output
    costs when half of it is unsupervised.

    Controls already measured on three seeds: AC_attn4_bce 0.3687 +/- 0.0111 for
    classification, AC_attn4_softthr 0.6935 +/- 0.0065 for regression.
    """
    for weight in (0.0, 0.1, 0.3):
        tag = f'{weight:g}'.replace('.', '')
        yield (f'D1_struct_aux{tag}',
               {'model': 'atom_jepa_tdi', 'model.tdi_head': 'true',
                'model.aux_pic50': weight}, ['classification'])
        yield (f'D2_multi_aux{tag}',
               {'model': 'atom_jepa_tdi', 'model.tdi_head': 'false',
                'model.aux_pic50': weight, 'train.loss': 'soft_threshold'},
               ['regression'])


def tdi_push_stage():
    """Two things the first Direction-D grid left open.

    It came out negative: the structured logit cost 0.017 MCC against an unstructured head
    (0.3520 +/- 0.0022 against 0.3687 +/- 0.0111, about 2.6 standard errors), and on
    regression the auxiliary pIC50 supervision did literally nothing -- aux=0 scored 0.6900
    and aux=0.3 scored 0.6899. The regression half is settled and is not re-run.

    The classification half is not, for two reasons. First, the auxiliary weight was still
    improving at the top of the range tried (0.3 was both the best and by far the most
    stable), so the grid may simply have stopped too early. Second, every Direction-D run
    used the default readout, while the classification task now has a better one
    (BEST_HEAD), so part of the 0.017 gap may be the head rather than the structure.

    Worth measuring even if the accuracy gap does not close: the structured head's spread
    across seeds is five times tighter than the control's (+/- 0.0022 against +/- 0.0111)
    and it treats the two enzymes evenly (0.3504 / 0.3537 against 0.3565 / 0.3810), which
    is a real property of the design and not an accident of one weight.
    """
    base = {'model': 'atom_jepa_tdi', 'model.tdi_head': 'true'}
    head = BEST_HEAD['classification']
    for weight in (1.0, 3.0):
        tag = f'{weight:g}'
        yield (f'D3_struct_aux{tag}', {**base, 'model.aux_pic50': weight},
               ['classification'])
    for weight in (0.3, 1.0):
        tag = f'{weight:g}'.replace('.', '')
        yield (f'D4_besthead_aux{tag}', {**base, **head, 'model.aux_pic50': weight},
               ['classification'])

STAGES = {'layers': layers_stage, 'degrees': degrees_stage, 'pool': pool_stage,
          'head': head_stage, 'loss': loss_stage, 'combine': combine_stage,
          'tdi': tdi_stage, 'tdi_push': tdi_push_stage, 'polish': polish_stage}


def run(job, env, dry_run):
    name, task, seed, overrides = job
    # A stage may select a different model by yielding a bare `model` override; every
    # stage that does not keeps the cached encoder, which is what makes a sweep affordable.
    overrides = dict(overrides)
    model = overrides.pop('model', 'atom_jepa_cached')
    command = [sys.executable, '-u', 'finetune.py', f'model={model}',
               f'task={task}', f'seed={seed}', 'device=cpu', 'eval_test=false',
               f'model.name={name}_s{seed}'] + [f'{k}={v}' for k, v in overrides.items()]
    if dry_run:
        print(' '.join(command))
        return name, task, seed, 0
    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        print(f'FAILED {name} {task} seed={seed}\n{result.stdout[-1500:]}\n'
              f'{result.stderr[-1500:]}', flush=True)
    else:
        print(f'ok {name:24s} {task:14s} seed={seed} {time.time() - start:5.0f}s', flush=True)
    return name, task, seed, result.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=sorted(STAGES))
    p.add_argument('--jobs', type=int, default=5, help='Runs in parallel')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    p.add_argument('--threads', type=int, default=2, help='Torch threads per run')
    p.add_argument('--base', nargs='*', default=[], metavar='KEY=VALUE',
                   help='Overrides every run in this stage carries, e.g. model.layers=[4]')
    p.add_argument('--random-seed', type=int, default=0, help='For the head search')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.jobs < 1 or args.seeds < 1:
        p.error('--jobs and --seeds must be positive')

    base = {}
    for item in args.base:
        if '=' not in item:
            p.error(f'--base takes KEY=VALUE, got {item!r}')
        key, value = item.split('=', 1)
        base[key] = value

    maker = STAGES[args.stage]
    configs = list(maker(random.Random(args.random_seed)) if args.stage == 'head' else maker())
    # A stage entry is (name, overrides) for both tasks, or (name, overrides, tasks).
    configs = [entry if len(entry) == 3 else (*entry, TASKS) for entry in configs]
    jobs = [(name, task, seed, {**base, **overrides})
            for (name, overrides, tasks), seed in itertools.product(configs, range(args.seeds))
            for task in tasks if task in args.tasks]
    if not jobs:
        p.error(f'stage {args.stage} has nothing to run for tasks {args.tasks}')
    print(f'{len(configs)} configurations over {args.seeds} seeds = {len(jobs)} runs, '
          f'{args.jobs} at a time', flush=True)

    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads),
           'MKL_NUM_THREADS': str(args.threads)}
    start = time.time()
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda j: run(j, env, args.dry_run), jobs))
    failed = [r for r in results if r[3]]
    print(f'\n{len(results) - len(failed)}/{len(results)} runs finished in '
          f'{(time.time() - start) / 60:.1f} min', flush=True)
    if failed:
        print('Failed: ' + ', '.join(f'{n}/{t}/s{s}' for n, t, s, _ in failed))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
