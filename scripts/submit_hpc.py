#!/usr/bin/env python3
"""Submit a Direction-B grid to the DTU HPC queue, one bsub per run.

Every job goes through finetune.sh, so there is a single definition of the environment,
the CUDA check and the output layout. Nothing is submitted until --submit is given: the
default prints the commands so they can be read first.

    python scripts/submit_hpc.py depth --queue gpua100 --precision bf16
    python scripts/submit_hpc.py depth --seeds 3 --only B1_full,B3_lpft --submit
    python scripts/submit_hpc.py control --submit

Runs are named B<n>_<what>_s<seed> and land in runs/<task>/<name>/<timestamp>/, which is
what scripts/aggregate_results.py groups over. Exploration jobs carry eval_test=false, so
the held-out labels stay untouched until a configuration has been chosen on validation.
"""
import argparse
import shlex
import subprocess
import sys

TASKS = ['regression', 'classification']
# Nothing in a run's name or overrides may contain a comma: bsub -env splits variables on it.
COSINE = 'train.scheduler=warmup_cosine'

# How much of the encoder to adapt, and how. B1 is the starter's recipe; the learning-rate
# ladder on it exists so the later rows are not just reporting a better encoder_lr.
DEPTH = [
    ('B0_frozen',      'model.unfreeze_blocks=0 train.epochs=60 model.readout_lr=1.0e-3'),
    ('B1_full_lr3e6',  'model.encoder_lr=3.0e-6'),
    ('B1_full',        ''),
    ('B1_full_lr3e5',  'model.encoder_lr=3.0e-5'),
    ('B2_full_cosine', COSINE),
    ('B3_lpft',        f'model.probe_epochs=5 train.epochs=25 {COSINE}'),
    ('B4_llrd065',     f'model.llrd=0.65 {COSINE}'),
    ('B4_llrd085',     f'model.llrd=0.85 {COSINE}'),
    ('B5_last2',       f'model.unfreeze_blocks=2 {COSINE}'),
    ('B5_last4',       f'model.unfreeze_blocks=4 {COSINE}'),
    ('B6_l2sp1e3',     f'model.l2sp=1.0e-3 {COSINE}'),
    ('B6_l2sp1e2',     f'model.l2sp=1.0e-2 {COSINE}'),
]

# The two controls the comparison needs to mean anything: the same architecture without the
# pretrained weights, and the pretrained encoder with an untrained head left frozen.
CONTROL = [
    ('B7_scratch',      f'model.pretrained=false model.encoder_lr=3.0e-5 {COSINE}'),
    ('B7_scratch_full', f'model.pretrained=false {COSINE}'),
]

# The three directions crossed on one model: the readout that won Direction A, the full
# fine-tuning that won Direction B, and the objective that won Direction C. Until now the
# best readout and the best adaptation lived in different code paths and had never met.
# F1 isolates the readout so the table can say which lever carries the result; the loss and
# pos_weight rows are task-specific, so submit this stage one task at a time with --only.
FUSION = [
    ('F1_full_attn4',         'model.pool=attention'),
    ('F2_full_attn4_softthr', 'model.pool=attention train.loss=soft_threshold'),
    ('F2_full_attn4_posw3',   'model.pool=attention train.pos_weight=3.0'),
    ('F3_full_attn4_lpft',    f'model.pool=attention model.probe_epochs=5 train.epochs=25 '
                              f'{COSINE}'),
    # The one lever the frozen winner had that the fine-tuned model lacked: reading every
    # block instead of only the top one. Weaker a priori here, since a trainable top block
    # can reshape itself, which is exactly why it is worth measuring rather than assuming.
    ('F4_mix_attn4',          'model.pool=attention model.layer_mix=scalar_mix'),
    ('F4_mix_attn4_softthr',  'model.pool=attention model.layer_mix=scalar_mix '
                              'train.loss=soft_threshold'),
]

STAGES = {'depth': DEPTH, 'control': CONTROL, 'fusion': FUSION}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=sorted(STAGES))
    # Measured on 2026-11-05 with `bqueues -u $USER`: this account may use gpuv100,
    # gpua100, gpua10, gpua40 and gpul40s, and nothing else — gpuh100 is refused outright.
    # gpua10 is the default because it was by far the shortest line (60 pending against
    # 1068 on gpua100, 1826 on gpuv100) and an A10 still holds this 39M-parameter encoder
    # with room to spare at batch size 8.
    p.add_argument('--queue', default='gpua10',
                   choices=['gpua10', 'gpua40', 'gpua100', 'gpul40s', 'gpuv100'],
                   help='gpua10 is usually the shortest queue; check with bqueues -u $USER')
    p.add_argument('--precision', default='bf16', choices=['fp32', 'bf16'],
                   help='bf16 is native on a10/a40/a100/l40s; a V100 emulates it and is slower')
    p.add_argument('--walltime', default='08:00')
    p.add_argument('--seeds', type=int, default=1,
                   help='1 to screen the grid; re-run the best few with 3')
    p.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    p.add_argument('--only', help='Comma-separated run names, to resubmit a subset')
    # Screening runs one seed, then the best few get replicates. Without this the second
    # pass would redo seed 0 and spend GPU time reproducing a number already in hand.
    p.add_argument('--seed-start', type=int, default=0, metavar='N',
                   help='First seed; --seed-start 1 --seeds 2 adds seeds 1 and 2')
    p.add_argument('--eval-test', action='store_true',
                   help='Only for the configurations already chosen on validation')
    p.add_argument('--submit', action='store_true', help='Actually call bsub')
    args = p.parse_args()
    if args.seeds < 1:
        p.error('--seeds must be positive')
    if args.seed_start < 0:
        p.error('--seed-start must not be negative')

    configs = STAGES[args.stage]
    if args.only:
        wanted = {n.strip() for n in args.only.split(',')}
        unknown = wanted - {name for name, _ in configs}
        if unknown:
            p.error(f'not in stage {args.stage}: {", ".join(sorted(unknown))}')
        configs = [c for c in configs if c[0] in wanted]

    submitted = 0
    for name, overrides in configs:
        for task in args.tasks:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                run = f'{name}_s{seed}'
                flags = (f'model.name={run} seed={seed} model.precision={args.precision} '
                         f'eval_test={str(args.eval_test).lower()} {overrides}').split()
                env = f'MODEL=atom_jepa_ft, TASK={task}, OVERRIDES={" ".join(flags)}'
                command = ['bsub', '-q', args.queue, '-J', f'{run}_{task[:3]}',
                           '-W', args.walltime, '-env', env]
                if args.submit:
                    with open('finetune.sh') as script:
                        result = subprocess.run(command, stdin=script, text=True)
                    if result.returncode:
                        print(f'bsub failed for {run} {task}', file=sys.stderr)
                        return 1
                else:
                    print(' '.join(shlex.quote(c) for c in command) + ' < finetune.sh')
                submitted += 1
    verb = 'submitted' if args.submit else 'would submit'
    print(f'\n{verb} {submitted} jobs to {args.queue} '
          f'({len(configs)} configurations x {len(args.tasks)} tasks x {args.seeds} seeds, '
          f'seeds {args.seed_start}..{args.seed_start + args.seeds - 1})')
    if not args.submit:
        print('Add --submit to send them.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
