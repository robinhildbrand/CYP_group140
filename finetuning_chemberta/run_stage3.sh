#!/bin/bash
#BSUB -q gpua100
#BSUB -J chemberta_stage3_optuna
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 02:00
#BSUB -o outputs/stage3_%J.out
#BSUB -e outputs/stage3_%J.err

set -euo pipefail
cd "${LS_SUBCWD}"
mkdir -p outputs
source .venv/bin/activate

python finetune_optuna_chemberta.py --task classification --n-trials 20
python finetune_optuna_chemberta.py --task regression --n-trials 15