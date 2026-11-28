#!/usr/bin/env python3
"""Fine-tune one model (Atom-JEPA or a baseline) for regression OR classification.

Configured with Hydra (conf/), e.g. python finetune.py model=morgan_mlp task=classification"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from models import MODELS
from models.gbm import LightGBMModel
from prepare_data import REG, CLS


def masked_loss(pred, y, task, mean, std, options, pos_weight=None, bounds=None):
    """Loss over the observed labels only. `options` is cfg.train.

    `loss='default'` is MSE on standardized targets for regression and plain BCE for
    classification — exactly what the starter trains with. The alternatives exist because
    neither default matches the metric its task is scored on:

    - regression is scored with ST-RAE, which counts only the distance falling OUTSIDE the
      credible interval of each label. `soft_threshold` optimizes that distance itself;
      `huber` and `l1` at least drop MSE's quadratic emphasis on the worst residuals.
    - classification is scored with MCC at a fixed 0.5 threshold on about 20% positives,
      where BCE is quite content to leave nearly every probability below that threshold.
      `pos_weight` raises the price of a missed positive.
    """
    mask = torch.isfinite(y)
    if not mask.any():
        return None
    kind = options.loss
    # Index BEFORE applying the loss: multiplying NaN losses by zero is not safe.
    if task == 'classification':
        if kind != 'default':
            raise ValueError(f'loss={kind!r} is a regression loss')
        weight = None if pos_weight is None else pos_weight.expand_as(y)[mask]
        return nn.functional.binary_cross_entropy_with_logits(pred[mask], y[mask],
                                                              pos_weight=weight)
    standardized, p = ((y - mean) / std)[mask], pred[mask]
    if kind == 'default':
        return nn.functional.mse_loss(p, standardized)
    if kind == 'huber':
        return nn.functional.huber_loss(p, standardized, delta=options.huber_delta)
    if kind == 'l1':
        return nn.functional.l1_loss(p, standardized)
    if kind != 'soft_threshold':
        raise ValueError(f'Unknown loss {kind!r}')
    if bounds is None:
        raise ValueError('loss=soft_threshold needs the credible intervals in the batch, '
                         'which this model does not provide (see models/targets.py)')
    # ST-RAE is defined on the labels, so undo the standardization before measuring.
    label = (pred * std + mean)[mask]
    outside = (torch.relu(bounds[..., 0][mask] - label)
               + torch.relu(label - bounds[..., 1][mask])).mean()
    # A pure dead-zone loss gives no gradient at all to a prediction already inside the
    # interval; loss_aux keeps a weak pull towards the label itself.
    return outside + options.loss_aux * nn.functional.mse_loss(p, standardized)


def positive_weight(setting, train, targets, device):
    """Per-endpoint cost multiplier for the positive class, or None.

    'auto' is negatives/positives measured on the TRAINING labels only, the weight that
    makes the two classes contribute equally."""
    if setting in (None, 'none'):
        return None
    if setting == 'auto':
        ratios = [(train[t] == 0).sum() / max(1, (train[t] == 1).sum()) for t in targets]
    else:
        ratios = [float(setting)] * len(targets)
    print(f'pos_weight: {dict(zip(targets, [round(float(r), 2) for r in ratios]))}', flush=True)
    return torch.tensor(ratios, dtype=torch.float32, device=device)


class Schedule:
    """Multiplies every learning rate by a per-step factor and by its group's `lr_scale`.

    Each group keeps the base rate it was created with, so a model that hands out one rate
    per layer keeps those ratios. `lr_scale` defaults to 1 and is owned by the model: it is
    how a strategy that switches a group off and on mid-training (probe-then-fine-tune,
    gradual unfreezing) does so without rebuilding the optimizer and losing its moments.
    With scheduler='none' and no lr_scale the factor is exactly 1 at every step, which is
    what every config that predates this class gets."""

    def __init__(self, optimizer, kind, total_steps, warmup_frac):
        self.optimizer, self.kind = optimizer, kind
        self.base = [group['lr'] for group in optimizer.param_groups]
        self.total = max(1, total_steps)
        self.warmup = min(self.total - 1, max(0, round(self.total * warmup_frac)))
        if kind not in ('none', 'warmup_cosine', 'warmup_constant'):
            raise ValueError(f'Unknown scheduler {kind!r}')

    def factor(self, step):
        if self.kind == 'none':
            return 1.
        if step < self.warmup:
            return (step + 1) / (self.warmup + 1)
        if self.kind == 'warmup_constant':
            return 1.
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))

    def apply(self, step):
        factor = self.factor(step)
        for group, base in zip(self.optimizer.param_groups, self.base):
            group['lr'] = base * group.get('lr_scale', 1.) * factor


def metrics(frame, predictions, targets, task):
    result = {}
    for j, target in enumerate(targets):
        y = frame[target].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y, pred = y[mask], predictions[mask, j]
        if not len(y):
            result[target] = {'n': 0, ('ST-RAE' if task == 'regression' else 'MCC'): None}
            continue
        if task == 'regression':
            low = frame[target + '_conf_low'].to_numpy(dtype=float)[mask]
            high = frame[target + '_conf_high'].to_numpy(dtype=float)[mask]
            if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
                raise ValueError(f'Invalid credible intervals for {target}')
            distance = lambda p: np.maximum(low - p, 0) + np.maximum(p - high, 0)
            denom = distance(y.mean()).sum()
            result[target] = {'MAE': float(mean_absolute_error(y, pred)),
                              'R2': float(r2_score(y, pred)) if len(y) > 1 else None,
                              'ST-RAE': float(distance(pred).sum() / denom) if denom > 0 else None,
                              'n': len(y)}
        else:
            both = len(np.unique(y)) == 2
            result[target] = {'MCC': float(matthews_corrcoef(y, pred >= .5)),
                              'AUROC': float(roc_auc_score(y, pred)) if both else None,
                              'AP': float(average_precision_score(y, pred)) if both else None,
                              'n': len(y), 'positive_fraction': float(y.mean())}
    primary = 'ST-RAE' if task == 'regression' else 'MCC'
    scores = [result[t][primary] for t in targets]
    # Do not silently omit an endpoint when reporting a macro score.
    result['macro_' + primary] = (float(np.mean(scores))
        if all(s is not None and np.isfinite(s) for s in scores) else None)
    return result


@torch.no_grad()
def predict(model, loader, task, mean, std):
    model.eval()
    out = []
    for batch in loader:
        p = model(batch)
        p = p * std + mean if task == 'regression' else p.sigmoid()
        out.append(p.float().cpu().numpy())
    values = np.concatenate(out)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite predictions')
    return values


def train_torch(cfg, train, val, targets, task, data, out):
    """Trains a torch model with the epoch loop; returns a function frame -> predictions."""
    if cfg.train.epochs < 1 or cfg.train.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    model = MODELS[cfg.model.type](cfg.model, cfg.readout, len(targets), data, cfg.device)
    model.fit_preprocessing(train)  # Feature scaling etc. is fitted on the training split ONLY.
    mean = torch.zeros(len(targets), device=model.device)
    std = torch.ones_like(mean)
    if task == 'regression':
        # Fit scaling on the training fold ONLY, separately for each observed endpoint.
        mean = torch.tensor(train[targets].mean().to_numpy(), dtype=torch.float32, device=model.device)
        std = torch.tensor(train[targets].std(ddof=0).to_numpy(), dtype=torch.float32,
                           device=model.device).clamp_min(1e-6)
    batch_size = cfg.train.batch_size
    train_loader = model.loader(train, targets, batch_size, shuffle=True)
    val_loader = model.loader(val, targets, batch_size)
    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)
    schedule = Schedule(optimizer, cfg.train.scheduler, cfg.train.epochs * len(train_loader),
                        cfg.train.warmup_frac)
    # Optional model hooks: a phase schedule over the epochs, and a regularizer the model
    # defines on its own weights. Models without them are unaffected.
    set_epoch = getattr(model, 'set_epoch', None)
    penalty = getattr(model, 'penalty', None)
    pos_weight = (positive_weight(cfg.train.pos_weight, train, targets, model.device)
                  if task == 'classification' else None)
    wants_bounds = task == 'regression' and cfg.train.loss == 'soft_threshold'
    step = 0
    best = float('inf')
    for epoch in range(1, cfg.train.epochs + 1):
        if set_epoch is not None:
            set_epoch(epoch, optimizer)
        model.train()
        loss_sum, observed = 0., 0
        for batch in train_loader:
            y = batch['y'].to(model.device)
            bounds = batch.get('bounds') if wants_bounds else None
            loss = masked_loss(model(batch).float(), y, task, mean, std, cfg.train, pos_weight,
                               None if bounds is None else bounds.to(model.device))
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            objective = loss
            if penalty is not None:
                extra = penalty()
                # Reported train_loss stays the data term, so histories remain comparable
                # across regularizer strengths.
                objective = loss if extra is None else loss + extra
            schedule.apply(step)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                     cfg.train.grad_clip)
            optimizer.step()
            step += 1
            n = int(torch.isfinite(y).sum())
            loss_sum += loss.item() * n
            observed += n
        score = metrics(val, predict(model, val_loader, task, mean, std), targets, task)
        primary = score['macro_ST-RAE' if task == 'regression' else 'macro_MCC']
        if primary is None:
            raise ValueError('Validation metric undefined; check endpoint coverage and split')
        selection = primary if task == 'regression' else -primary
        record = {'epoch': epoch, 'train_loss': loss_sum / observed, **score}
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if selection < best:
            best = selection
            torch.save({'model': model.state_dict(), 'mean': mean, 'std': std,
                        'epoch': epoch, 'config': cfg_dict(cfg)}, out / 'best.pt')
    saved = torch.load(out / 'best.pt', map_location=model.device, weights_only=True)
    model.load_state_dict(saved['model'])
    mean, std = saved['mean'], saved['std']
    print(f'Best epoch: {saved["epoch"]}', flush=True)
    return lambda frame: predict(model, model.loader(frame, targets, batch_size), task, mean, std)


def cfg_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg):
    if cfg.task not in ('regression', 'classification'):
        raise ValueError(f'Unknown task {cfg.task!r}')
    out = Path(HydraConfig.get().runtime.output_dir)
    data = Path(cfg.data)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    task = cfg.task
    targets = REG if task == 'regression' else CLS
    train_all = pd.read_csv(data / task / 'train.csv')
    val_all = pd.read_csv(data / task / 'val.csv')
    frame = pd.concat([train_all, val_all], ignore_index=True)
    train = train_all[train_all[targets].notna().any(axis=1)].copy()
    val = val_all[val_all[targets].notna().any(axis=1)].copy()
    if not len(train) or not len(val):
        raise ValueError('Empty train or validation set')
    if set(train_all.canonical_smiles) & set(val_all.canonical_smiles):
        raise ValueError('Exact molecule leakage between train and validation')
    for t in targets:
        if not train[t].notna().any() or not val[t].notna().any():
            raise ValueError(f'Missing train/validation labels for {t}')
        if task == 'classification':
            if not set(frame[t].dropna()).issubset({0, 1}):
                raise ValueError(f'Nonbinary labels: {t}')
            if train[t].nunique() != 2 or val[t].nunique() != 2:
                raise ValueError(f'{t} needs both classes in train and validation; revise split')
    config = cfg_dict(cfg)
    packages = ['torch', 'atom-jepa', 'rdkit', 'numpy', 'pandas', 'scikit-learn', 'hydra-core',
                'descriptastorus', 'lightgbm']
    config.update(targets=targets, output_dir=str(out),
                  versions={k: importlib.metadata.version(k) for k in packages},
                  preparation=json.loads((data / 'manifest.json').read_text()),
                  split_sha256={name: hashlib.sha256((data / task / f'{name}.csv').read_bytes()).hexdigest()
                                for name in ['train', 'val', 'test']})
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    frame[['Molecule_Name', 'key', 'split']].to_csv(out / 'split.csv', index=False)
    # Sanity baseline fitted on training labels only.
    baseline = np.tile(train[targets].mean().to_numpy(), (len(val), 1))
    (out / 'baseline_metrics.json').write_text(json.dumps(metrics(val, baseline, targets, task), indent=2))
    if cfg.model.type == 'lightgbm':
        model = LightGBMModel(cfg.model, data, cfg.seed)
        best = model.fit(train, val, targets, task, out)
        (out / 'best_iterations.json').write_text(json.dumps(best, indent=2))
        predict_frame = model.predict
    else:
        predict_frame = train_torch(cfg, train, val, targets, task, data, out)
    validation_predictions = predict_frame(val)
    (out / 'validation_metrics.json').write_text(json.dumps(metrics(val, validation_predictions, targets, task), indent=2))
    vp = val[['SMILES', 'Molecule_Name']].copy()
    vp[targets] = validation_predictions
    vp.to_csv(out / 'validation_predictions.csv', index=False)
    # Exploration runs stop here so the held-out labels are never even read. Only the
    # configurations already selected on validation are run again with eval_test=true.
    if not cfg.eval_test:
        print('eval_test=false: the test split was not read', flush=True)
        return
    # Only now load and evaluate the held-out labelled test split.
    test = pd.read_csv(data / task / 'test.csv')
    if not len(test):
        raise ValueError('Empty test set')
    if set(frame.canonical_smiles) & set(test.canonical_smiles):
        raise ValueError('Exact molecule leakage into the test split')
    predictions = predict_frame(test)
    test_scores = metrics(test, predictions, targets, task)
    (out / 'test_metrics.json').write_text(json.dumps(test_scores, indent=2))
    # Classification exports probabilities; metrics use a fixed 0.5 threshold.
    output = test[['SMILES', 'Molecule_Name']].copy()
    output[targets] = predictions
    output.to_csv(out / 'test_predictions.csv', index=False)
    print(f'Held-out test metrics: {test_scores}')

if __name__ == '__main__':
    main()
