"""
Entry point for training and evaluating a model on MERT embeddings.

Usage examples:
    python run.py                            # MLP with defaults
    python run.py --hidden_dim 256           # narrower MLP
    python run.py --scheduler plateau        # swap scheduler
    python run.py --n_epochs 500 --lr 5e-4  # quick sweep

To add a new architecture, create config/<arch>.py + models/<arch>.py,
register it in models/__init__.py, and pass --model_name <arch>.
"""

import argparse
import dataclasses
import random
from pathlib import Path
import numpy as np
import torch

from config.mlp import MLPConfig
from config.cosine_arcface import CosineArcFaceConfig
from data import load_data
from models import build_model
from train import build_criterion, train, evaluate
from evaluate import (
    plot_training_curves,
    plot_per_class_accuracy,
    plot_confusion_matrix,
    print_final_summary,
)

# ── Config registry ───────────────────────────────────────────────────────────
# Add new config classes here as you introduce new architectures.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on MERT embeddings.")

    # Which model / config to use.
    parser.add_argument("--model_name", type=str, default="mlp",
                        choices=list(CONFIG_REGISTRY.keys()))

    # Allow any BaseConfig or subclass field to be overridden from the CLI.
    # We parse these as key=value strings and apply them after building the config.
    parser.add_argument("--hidden_dim",    type=int,   default=None)
    parser.add_argument("--embed_dim",     type=int,   default=None)
    parser.add_argument("--dropout_p",     type=float, default=None)
    parser.add_argument("--activation",     type=str,   default=None,
                        choices=["gelu", "relu"])
    parser.add_argument("--use_layer_norm", type=str2bool, default=None)
    parser.add_argument("--head_type",      type=str,   default=None,
                        choices=["softmax", "cosine", "arcface"])
    parser.add_argument("--scale",          type=float, default=None)
    parser.add_argument("--margin",         type=float, default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--weight_decay",  type=float, default=None)
    parser.add_argument("--n_epochs",      type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--scheduler",     type=str,   default=None,
                        choices=["cosine", "plateau"])
    parser.add_argument("--optimizer",     type=str,   default=None,
                        choices=["adam", "adamw"])
    parser.add_argument("--seed",          type=int,   default=None)
    parser.add_argument("--val_ratio",     type=float, default=None)
    parser.add_argument("--split_seed",    type=int,   default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--save_best_only", type=str2bool, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Apply any non-None CLI args onto the config object."""
    for field in dataclasses.fields(cfg):
        val = getattr(args, field.name, None)
        if val is not None:
            if field.name in {"embeddings_dir", "output_dir"}:
                val = Path(val)
            setattr(cfg, field.name, val)
    return cfg


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Build config, then apply CLI overrides.
    cfg = CONFIG_REGISTRY[args.model_name]()
    cfg = apply_overrides(cfg, args)

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(f"Run:    {cfg.run_name}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, le, d, n_classes, y_train_np = load_data(
        cfg.embeddings_dir,
        cfg.batch_size,
        val_ratio=cfg.val_ratio,
        split_seed=cfg.split_seed,
        include_val_in_train=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, d=d, n_classes=n_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model}")
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_criterion(y_train_np, n_classes, cfg.label_smoothing, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_out = train(model, train_loader, val_loader, cfg, criterion, monitor_name="val")
    history = train_out["history"]

    # ── Save checkpoint + config ──────────────────────────────────────────────
    ckpt_path = cfg.output_dir / f"{cfg.run_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "best_val_top1": train_out["best_metric"],
            "best_epoch": train_out["best_epoch"],
        },
               ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    final = evaluate(model, test_loader, criterion, device)
    y_pred = final["preds"].numpy()
    y_true = final["labels"].numpy()

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history, cfg, n_classes)
    accs_all = plot_per_class_accuracy(y_pred=y_pred, y_true=y_true,
                                        le=le, final_top1=final["top1"], cfg=cfg)
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, le=le, cfg=cfg)
    print_final_summary(final, cfg, n_classes, accs_all)


if __name__ == "__main__":
    main()
