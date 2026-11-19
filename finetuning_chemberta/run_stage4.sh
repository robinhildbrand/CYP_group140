#!/bin/bash
#BSUB -q gpua100
#BSUB -J chemberta_stage4_test
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 01:00
#BSUB -o outputs/stage4_%J.out
#BSUB -e outputs/stage4_%J.err

set -euo pipefail
cd "${LS_SUBCWD}"
mkdir -p outputs
source .venv/bin/activate

# 1. Final Classification Evaluation (Stage 3 Trial #14 winner, 3 seeds, held-out test set)
for SEED in 0 1 2; do
    RUN_NAME="S4_cls_final_seed${SEED}"
    python finetune.py \
        model=chemberta \
        readout=mlp \
        task=classification \
        seed=${SEED} \
        eval_test=true \
        model.name=${RUN_NAME} \
        model.checkpoint=DeepChem/ChemBERTa-77M-MTR \
        model.pool=cls \
        model.freeze_encoder=true \
        model.readout_lr=5.656130e-04 \
        train.weight_decay=4.697116e-04 \
        train.scheduler=none \
        train.pos_weight=none \
        readout.hidden_dim=256 \
        readout.dropout=0.10 \
        hydra.run.dir=runs/classification/${RUN_NAME}
done

# 2. Final Regression Evaluation (Stage 3 Trial #14 winner, 3 seeds, held-out test set)
for SEED in 0 1 2; do
    RUN_NAME="S4_reg_final_seed${SEED}"
    python finetune.py \
        model=chemberta \
        readout=mlp \
        task=regression \
        seed=${SEED} \
        eval_test=true \
        model.name=${RUN_NAME} \
        model.checkpoint=DeepChem/ChemBERTa-77M-MTR \
        model.pool=cls \
        model.freeze_encoder=false \
        model.encoder_lr=3.730682e-05 \
        model.readout_lr=9.491629e-05 \
        train.weight_decay=1.017734e-03 \
        train.scheduler=warmup_cosine \
        train.loss=soft_threshold \
        train.loss_aux=0.194 \
        readout.hidden_dim=128 \
        readout.dropout=0.35 \
        hydra.run.dir=runs/regression/${RUN_NAME}
done