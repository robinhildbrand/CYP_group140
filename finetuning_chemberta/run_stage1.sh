#!/bin/bash
#BSUB -q gpua100
#BSUB -J chemberta_stage1
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 01:30
#BSUB -o outputs/stage1_%J.out
#BSUB -e outputs/stage1_%J.err

set -euo pipefail
cd "${LS_SUBCWD}"
mkdir -p outputs
source .venv/bin/activate

# Stage 1: Compare Property-Informed (MTR) vs. Masked LM (MLM) checkpoints
# and Frozen vs. End-to-End encoder training across both CYP tasks (seed=0).
for TASK in regression classification; do
    for CKPT_TAG in MTR MLM; do
        CKPT="DeepChem/ChemBERTa-77M-${CKPT_TAG}"
        for FREEZE in true false; do
            MODE_TAG=$([ "$FREEZE" = "true" ] && echo "frozen" || echo "e2e")
            RUN_NAME="S1_${TASK[:3]}_${CKPT_TAG}_${MODE_TAG}"

            python finetune.py \
                model=chemberta \
                task=${TASK} \
                seed=0 \
                eval_test=false \
                model.name=${RUN_NAME} \
                model.checkpoint=${CKPT} \
                model.freeze_encoder=${FREEZE} \
                hydra.run.dir=runs/${TASK}/${RUN_NAME}
        done
    done
done