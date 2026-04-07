"""
Manual random search for cosine/arcface hyperparameters.

Example:
    python search.py --model_name cosine_arcface --n_trials 40 --seed 42
"""

import argparse
import copy
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from config.cosine_arcface import CosineArcFaceConfig
from config.mlp import MLPConfig
from data import load_data
from models import build_model
from train import build_criterion, train, evaluate


CONFIG_REGISTRY = {
    "mlp": MLPConfig,
    "cosine_arcface": CosineArcFaceConfig,
}


def str2bool(value: str) -> bool:
    v = value.lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Random hyperparameter search on MERT models.")
    parser.add_argument("--model_name", type=str, default="cosine_arcface",
                        choices=list(CONFIG_REGISTRY.keys()))
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="val_top1", choices=["val_top1"])
    parser.add_argument("--sampler", type=str, default="random", choices=["random"])
    parser.add_argument("--output_subdir", type=str, default="search")

    # Shared training knobs.
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cosine", "plateau"])
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--save_best_only", type=str2bool, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def apply_overrides(cfg, args):
    for field in dataclasses.fields(cfg):
        val = getattr(args, field.name, None)
        if val is not None:
            if field.name in {"embeddings_dir", "output_dir"}:
                val = Path(val)
            setattr(cfg, field.name, val)
    return cfg


def sample_cosine_arcface_hparams(cfg, rng: random.Random):
    cfg.head_type = rng.choice(["cosine", "arcface"])
    cfg.hidden_dim = rng.choice([256, 512])
    cfg.embed_dim = rng.choice([128, 256])
    cfg.dropout_p = rng.uniform(0.1, 0.3)
    cfg.activation = rng.choice(["gelu", "relu"])
    cfg.lr = rng.choice([1e-3, 3e-4, 5e-4])
    cfg.weight_decay = rng.choice([1e-4, 3e-4, 1e-3])
    cfg.scale = rng.uniform(16.0, 32.0)
    if cfg.head_type == "arcface":
        cfg.margin = rng.uniform(0.2, 0.35)
    else:
        cfg.margin = 0.25
    cfg.optimizer = "adamw"
    return cfg


def run_trial(cfg):
    device = torch.device(cfg.device)
    train_loader, val_loader, test_loader, le, d, n_classes, y_train_np = load_data(
        cfg.embeddings_dir,
        cfg.batch_size,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
        include_val_in_train=False,
    )
    if val_loader is None:
        raise ValueError("Search requires a validation split. Set val_ratio > 0.")

    model = build_model(cfg, d=d, n_classes=n_classes)
    criterion = build_criterion(y_train_np, n_classes, cfg.label_smoothing, device)
    train_out = train(model, train_loader, val_loader, cfg, criterion, monitor_name="val")
    val_metrics = evaluate(model, val_loader, criterion, device)

    return {
        "model": model,
        "criterion": criterion,
        "n_classes": n_classes,
        "train_out": train_out,
        "val_metrics": val_metrics,
        "label_encoder": le,
        "test_loader": test_loader,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    cfg_cls = CONFIG_REGISTRY[args.model_name]
    base_cfg = apply_overrides(cfg_cls(), args)

    # Search output folder with timestamp for reproducibility.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_cfg.output_dir = Path(base_cfg.output_dir) / args.output_subdir / timestamp
    base_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = base_cfg.output_dir / "trials.jsonl"
    summary_json = base_cfg.output_dir / "summary.json"

    print(f"Starting search: model={args.model_name}, trials={args.n_trials}, seed={args.seed}")
    print(f"Artifacts: {base_cfg.output_dir}")

    best = None
    all_rows = []

    for trial_idx in range(1, args.n_trials + 1):
        cfg = copy.deepcopy(base_cfg)
        if cfg.model_name == "cosine_arcface":
            cfg = sample_cosine_arcface_hparams(cfg, rng)
        cfg.seed = args.seed + trial_idx

        set_seeds(cfg.seed)
        t0 = time.time()
        run_out = run_trial(cfg)
        duration = time.time() - t0

        val_top1 = float(run_out["val_metrics"]["top1"])
        val_top5 = float(run_out["val_metrics"]["top5"])
        val_loss = float(run_out["val_metrics"]["loss"])
        best_epoch = int(run_out["train_out"]["best_epoch"])
        best_metric = float(run_out["train_out"]["best_metric"])

        ckpt_path = cfg.output_dir / f"trial_{trial_idx:03d}_{cfg.run_name}.pt"
        torch.save(
            {
                "model_state": run_out["model"].state_dict(),
                "config": dataclasses.asdict(cfg),
                "best_val_top1": best_metric,
                "best_epoch": best_epoch,
            },
            ckpt_path,
        )

        row = {
            "trial": trial_idx,
            "seed": cfg.seed,
            "run_name": cfg.run_name,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "duration_sec": duration,
            "checkpoint": str(ckpt_path),
            "config": json_safe(dataclasses.asdict(cfg)),
        }
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        all_rows.append(row)
        if best is None or row["val_top1"] > best["val_top1"]:
            best = row

        print(
            f"[trial {trial_idx:03d}/{args.n_trials}] val_top1={val_top1:.4f} "
            f"val_top5={val_top5:.4f} loss={val_loss:.4f} "
            f"best_so_far={best['val_top1']:.4f}"
        )

    if best is None:
        raise RuntimeError("No completed trials.")

    print("\nSearch complete.")
    print(f"Best trial: {best['trial']} | val_top1={best['val_top1']:.4f}")
    best_cfg_dict = best["config"]
    best_cfg = cfg_cls(**best_cfg_dict)

    # Final protocol: retrain on train+val, evaluate once on test.
    print("\nRetraining best configuration on train+val...")
    set_seeds(best_cfg.seed)
    device = torch.device(best_cfg.device)
    train_loader, val_loader, test_loader, le, d, n_classes, y_train_np = load_data(
        best_cfg.embeddings_dir,
        best_cfg.batch_size,
        val_ratio=best_cfg.val_ratio,
        split_seed=best_cfg.split_seed,
        include_val_in_train=True,
    )
    model = build_model(best_cfg, d=d, n_classes=n_classes)
    criterion = build_criterion(y_train_np, n_classes, best_cfg.label_smoothing, device)
    train_out = train(model, train_loader, None, best_cfg, criterion, monitor_name="train")
    final_test = evaluate(model, test_loader, criterion, device)

    final_ckpt = best_cfg.output_dir / f"best_retrained_{best_cfg.run_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": dataclasses.asdict(best_cfg),
            "selected_from_trial": best["trial"],
            "selection_metric": "val_top1",
            "selected_val_top1": best["val_top1"],
            "final_test": {
                "top1": float(final_test["top1"]),
                "top5": float(final_test["top5"]),
                "loss": float(final_test["loss"]),
            },
            "best_epoch_retrain": int(train_out["best_epoch"]),
        },
        final_ckpt,
    )

    leaderboard = sorted(all_rows, key=lambda r: r["val_top1"], reverse=True)
    summary = {
        "search": {
            "model_name": args.model_name,
            "n_trials": args.n_trials,
            "metric": args.metric,
            "sampler": args.sampler,
            "seed": args.seed,
            "results_file": str(results_jsonl),
        },
        "best_trial": best,
        "top5_trials": leaderboard[:5],
        "final_retrain": {
            "checkpoint": str(final_ckpt),
            "test_top1": float(final_test["top1"]),
            "test_top5": float(final_test["top5"]),
            "test_loss": float(final_test["loss"]),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal retrain/test done.")
    print(f"Test top-1: {final_test['top1']:.4f}")
    print(f"Test top-5: {final_test['top5']:.4f}")
    print(f"Summary saved: {summary_json}")


if __name__ == "__main__":
    main()
