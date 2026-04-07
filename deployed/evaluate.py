from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def plot_training_curves(history: dict, cfg, n_classes: int) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    eval_prefix = "val" if "val_loss" in history else "test"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="steelblue")
    ax.plot(epochs, history[f"{eval_prefix}_loss"],  label=eval_prefix.title(),  color="coral", linestyle="--")
    ax.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="Loss Curve")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_top1"], label="Train top-1", color="steelblue")
    ax.plot(epochs, history[f"{eval_prefix}_top1"],  label=f"{eval_prefix.title()} top-1",  color="coral",     linestyle="--")
    ax.plot(epochs, history["train_top5"], label="Train top-5", color="royalblue", alpha=0.5)
    ax.plot(epochs, history[f"{eval_prefix}_top5"],  label=f"{eval_prefix.title()} top-5",  color="tomato",    alpha=0.5, linestyle="--")
    ax.axhline(1 / n_classes, color="gray", linestyle=":", label=f"Random ({1/n_classes:.3f})")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["lr"], color="mediumpurple")
    ax.set(xlabel="Epoch", ylabel="Learning Rate", title=f"LR Schedule ({cfg.scheduler})")
    ax.set_yscale("log"); ax.grid(alpha=0.3)

    fig.suptitle(cfg.run_name, fontsize=10)
    plt.tight_layout()
    _save(cfg.output_dir, cfg.run_name, "training_curves")
    plt.show()


def plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                             le: LabelEncoder, final_top1: float, cfg) -> None:
    per_class = {}
    for idx, name in enumerate(le.classes_):
        mask = y_true == idx
        if mask.sum() > 0:
            per_class[name] = (y_pred[mask] == idx).mean()

    sorted_cls  = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    accs_all    = list(per_class.values())

    print(f"Median per-class accuracy: {np.median(accs_all):.4f}")
    print(f"Mean per-class accuracy:   {np.mean(accs_all):.4f}")
    print(f"Classes with 100% acc:     {sum(1 for v in accs_all if v == 1.0)}")
    print(f"Classes with   0% acc:     {sum(1 for v in accs_all if v == 0.0)}")
    print("\nHardest pieces:")
    for name, acc in sorted_cls[-10:]:
        print(f"  {name:<40} {acc:.2f}")

    n_show = min(cfg.top_n_classes_chart, len(sorted_cls))
    half   = n_show // 2
    names  = [c for c, _ in sorted_cls]
    accs   = [a for _, a in sorted_cls]

    display_names = names[:half] + names[-half:]
    display_accs  = accs[:half]  + accs[-half:]
    colors        = ["#4CAF50"] * half + ["#F44336"] * half

    fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.3)))
    ax.barh(display_names, display_accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(final_top1, color="black", linestyle="--", linewidth=1.2,
               label=f"Overall top-1 ({final_top1:.2f})")
    ax.set(xlabel="Per-Class Accuracy",
           title=f"Per-Class Accuracy — Top {half} and Bottom {half} Pieces",
           xlim=(0, 1.05))
    ax.legend()
    ax.set_yticklabels([n.split("__")[-1] for n in display_names], fontsize=8)
    plt.tight_layout()
    _save(cfg.output_dir, cfg.run_name, "per_class_accuracy")
    plt.show()

    return accs_all


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           le: LabelEncoder, cfg) -> None:
    n_classes = len(le.classes_)
    cm        = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)

    flat_idx       = np.argsort(cm_errors.ravel())[::-1][:cfg.top_n_confused]
    top_true, top_pred = np.unravel_index(flat_idx, cm_errors.shape)
    involved       = sorted(set(top_true) | set(top_pred))
    sub_cm         = cm[np.ix_(involved, involved)]
    names_short    = [le.classes_[i].split("__")[-1] for i in involved]

    sz  = max(8, len(involved) * 0.6)
    fig, ax = plt.subplots(figsize=(sz, sz))
    sns.heatmap(sub_cm, xticklabels=names_short, yticklabels=names_short,
                annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix — {len(involved)} Most Confused Classes")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save(cfg.output_dir, cfg.run_name, "confusion_matrix")
    plt.show()

    print("Top confused pairs (true → predicted, count):")
    for ti, pi in zip(top_true, top_pred):
        count = cm[ti, pi]
        if count > 0:
            print(f"  {le.classes_[ti].split('__')[-1]:<30} → "
                  f"{le.classes_[pi].split('__')[-1]:<30} ({count} snippets)")


def print_final_summary(final_metrics: dict, cfg, n_classes: int,
                         accs_all: list) -> None:
    total_params = sum(p.numel() for p in [])  # placeholder; passed from run.py
    print("=" * 55)
    print(f"FINAL TEST RESULTS — {cfg.model_name.upper()}")
    print("=" * 55)
    print(f"  Top-1 Accuracy:  {final_metrics['top1']:.4f}  ({final_metrics['top1']*100:.1f}%)")
    print(f"  Top-5 Accuracy:  {final_metrics['top5']:.4f}  ({final_metrics['top5']*100:.1f}%)")
    print(f"  Test Loss:       {final_metrics['loss']:.4f}")
    print(f"  Median per-class:{np.median(accs_all):.4f}")
    print(f"  Random baseline: {1/n_classes:.4f}  ({100/n_classes:.2f}%)")
    print("=" * 55)


# ── Internal helper ───────────────────────────────────────────────────────────

def _save(output_dir: Path, run_name: str, plot_name: str) -> None:
    path = output_dir / f"{run_name}_{plot_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
