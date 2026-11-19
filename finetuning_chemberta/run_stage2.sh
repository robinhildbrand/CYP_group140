#!/bin/bash
#BSUB -q gpua100
#BSUB -J chemberta_stage2
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 01:30
#BSUB -o outputs/stage2_%J.out
#BSUB -e outputs/stage2_%J.err

set -euo pipefail
cd "${LS_SUBCWD}"
mkdir -p outputs
source .venv/bin/activate

CKPT="DeepChem/ChemBERTa-77M-MTR"

# --- Regression (End-to-End): [CLS] Pooling, soft_threshold Loss & Cosine Schedule ---
python finetune.py \
    model=chemberta task=regression seed=0 eval_test=false \
    model.name=S2_reg_cls_default \
    model.checkpoint=${CKPT} model.freeze_encoder=false model.pool=cls \
    hydra.run.dir=runs/regression/S2_reg_cls_default

python finetune.py \
    model=chemberta task=regression seed=0 eval_test=false \
    model.name=S2_reg_cls_soft_thresh \
    model.checkpoint=${CKPT} model.freeze_encoder=false model.pool=cls \
    train.loss=soft_threshold \
    hydra.run.dir=runs/regression/S2_reg_cls_soft_thresh

python finetune.py \
    model=chemberta task=regression seed=0 eval_test=false \
    model.name=S2_reg_cls_soft_thresh_cosine \
    model.checkpoint=${CKPT} model.freeze_encoder=false model.pool=cls \
    train.loss=soft_threshold train.scheduler=warmup_cosine \
    hydra.run.dir=runs/regression/S2_reg_cls_soft_thresh_cosine

# --- Classification (Frozen Encoder): [CLS] Pooling, Unweighted BCE & Cosine Schedule ---
python finetune.py \
    model=chemberta task=classification seed=0 eval_test=false \
    model.name=S2_cls_cls_default \
    model.checkpoint=${CKPT} model.freeze_encoder=true model.pool=cls \
    hydra.run.dir=runs/classification/S2_cls_cls_default

python finetune.py \
    model=chemberta task=classification seed=0 eval_test=false \
    model.name=S2_cls_cls_unweighted \
    model.checkpoint=${CKPT} model.freeze_encoder=true model.pool=cls \
    train.pos_weight=none \
    hydra.run.dir=runs/classification/S2_cls_cls_unweighted

python finetune.py \
    model=chemberta task=classification seed=0 eval_test=false \
    model.name=S2_cls_cls_unweighted_cosine \
    model.checkpoint=${CKPT} model.freeze_encoder=true model.pool=cls \
    train.pos_weight=none train.scheduler=warmup_cosine \
    hydra.run.dir=runs/classification/S2_cls_cls_unweighted_cosine