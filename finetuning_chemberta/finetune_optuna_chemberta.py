"""Stage 3 Bayesian hyperparameter optimization via Optuna wrapping finetune.py."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import optuna


def run_trial(task: str, trial: optuna.Trial, readouts: list[str]) -> float:
    run_name = f"S3_{task[:3]}_trial{trial.number:02d}"
    out_dir = Path("runs") / task / run_name

    readout_type = trial.suggest_categorical("readout", readouts)
    readout_lr = trial.suggest_float("readout_lr", 5e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["none", "warmup_cosine"])

    cmd = [
        sys.executable, "finetune.py",
        "model=chemberta",
        f"readout={readout_type}",
        f"task={task}",
        "seed=0",
        "eval_test=false",
        f"model.name={run_name}",
        "model.checkpoint=DeepChem/ChemBERTa-77M-MTR",
        "model.pool=cls",
        f"model.readout_lr={readout_lr:.6e}",
        f"train.weight_decay={weight_decay:.6e}",
        f"train.scheduler={scheduler}",
        f"hydra.run.dir={out_dir}",
    ]

    if readout_type in ("mlp", "resmlp"):
        dropout = trial.suggest_float("dropout", 0.0, 0.35, step=0.05)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        cmd += [
            f"readout.dropout={dropout:.2f}",
            f"readout.hidden_dim={hidden_dim}",
        ]

    if task == "regression":
        encoder_lr = trial.suggest_float("encoder_lr", 5e-6, 5e-5, log=True)
        loss_aux = trial.suggest_float("loss_aux", 0.02, 0.25)
        cmd += [
            "model.freeze_encoder=false",
            f"model.encoder_lr={encoder_lr:.6e}",
            "train.loss=soft_threshold",
            f"train.loss_aux={loss_aux:.3f}",
        ]
    else:
        cmd += [
            "model.freeze_encoder=true",
            "train.pos_weight=none",
        ]

    subprocess.run(cmd, check=True)

    metrics = json.loads((out_dir / "validation_metrics.json").read_text())
    score = metrics["macro_ST-RAE"] if task == "regression" else metrics["macro_MCC"]
    print(f"[Trial {trial.number:02d}] score={score:.4f} | params={trial.params}", flush=True)
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--n-trials", type=int, default=16)
    args = parser.parse_args()

    available_readouts = [
        r for r in ["linear", "mlp", "resmlp"]
        if Path(f"conf/readout/{r}.yaml").exists()
    ]

    Path("runs").mkdir(parents=True, exist_ok=True)
    direction = "minimize" if args.task == "regression" else "maximize"
    study = optuna.create_study(
        study_name=f"chemberta_{args.task}_s3",
        storage="sqlite:///runs/optuna_stage3.db",
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: run_trial(args.task, t, available_readouts), n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print(f"BEST {args.task.upper()} TRIAL (#{study.best_trial.number}): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    main()