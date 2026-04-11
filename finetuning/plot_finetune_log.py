"""
plot_finetune_log.py — Visualise training history from finetune_log_*.csv files.

Produces a 2×2 figure with four panels:
  1. Test Top-1 Accuracy  (with best-epoch marker and 34.4% baseline)
  2. Test Top-5 Accuracy
  3. Test Loss
  4. Learning Rate        (log scale)

Phase 1 and Phase 2 regions are shaded differently so the transition is obvious.
Multiple log files can be overlaid on the same axes for easy comparison.

Usage
-----
  # Single run
  python plot_finetune_log.py data/finetune/finetune_log_20240601_120000.csv

  # Compare several runs (each gets its own colour)
  python plot_finetune_log.py data/finetune/finetune_log_*.csv

  # Save to file instead of showing interactively
  python plot_finetune_log.py finetune_log_*.csv --out training_curves.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ── Aesthetics ────────────────────────────────────────────────────────────────

BASELINE_TOP1 = 0.344          # frozen MERT + logistic regression
PHASE_COLORS  = {"phase1": "#d0e8ff", "phase2": "#ffecd0"}   # background shading
LINE_COLORS   = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_log(path: Path) -> pd.DataFrame:
    """Load a finetune CSV log and add a global step column."""
    df = pd.read_csv(path)
    # Renumber epochs globally so Phase 2 continues after Phase 1
    df["global_epoch"] = df.groupby("phase").cumcount() + 1
    phase_sizes = df.groupby("phase")["epoch"].count()

    # Assign a global step: phase1 epochs come first, then phase2
    p1_len = phase_sizes.get("phase1", 0)
    def _global(row):
        if row["phase"] == "phase1":
            return row["epoch"]
        else:
            return p1_len + row["epoch"]

    df["step"] = df.apply(_global, axis=1)
    return df


def shade_phases(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw coloured background bands for each phase."""
    for phase, grp in df.groupby("phase"):
        ax.axvspan(
            grp["step"].min() - 0.5,
            grp["step"].max() + 0.5,
            color=PHASE_COLORS.get(phase, "#eeeeee"),
            alpha=0.35,
            zorder=0,
        )


def mark_best(ax: plt.Axes, df: pd.DataFrame, col: str, better: str = "max") -> None:
    """Place a star at the best epoch for *col*."""
    idx = df[col].idxmax() if better == "max" else df[col].idxmin()
    best_row = df.loc[idx]
    ax.plot(
        best_row["step"], best_row[col],
        marker="*", markersize=12, color="gold",
        markeredgecolor="black", markeredgewidth=0.6,
        zorder=5, linestyle="none",
    )
    ax.annotate(
        f"{best_row[col]:.3f}",
        xy=(best_row["step"], best_row[col]),
        xytext=(4, 4), textcoords="offset points",
        fontsize=7.5, color="black",
    )


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_logs(log_paths: list[Path], out_path: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Fine-tuning Training History", fontsize=14, fontweight="bold")

    ax_top1, ax_top5, ax_loss, ax_lr = axes.flat

    legend_patches = []

    for i, path in enumerate(log_paths):
        df    = load_log(path)
        color = LINE_COLORS[i % len(LINE_COLORS)]
        label = path.stem  # e.g. finetune_log_20240601_120000

        # ── Panel 1: Top-1 accuracy ───────────────────────────────────────
        shade_phases(ax_top1, df)
        ax_top1.plot(df["step"], df["test_top1"], color=color, linewidth=1.8, label=label)
        mark_best(ax_top1, df, "test_top1", better="max")

        # ── Panel 2: Top-5 accuracy ───────────────────────────────────────
        shade_phases(ax_top5, df)
        ax_top5.plot(df["step"], df["test_top5"], color=color, linewidth=1.8)
        mark_best(ax_top5, df, "test_top5", better="max")

        # ── Panel 3: Loss ─────────────────────────────────────────────────
        shade_phases(ax_loss, df)
        ax_loss.plot(df["step"], df["test_loss"], color=color, linewidth=1.8)
        mark_best(ax_loss, df, "test_loss", better="min")

        # ── Panel 4: Learning rate ────────────────────────────────────────
        shade_phases(ax_lr, df)
        ax_lr.plot(df["step"], df["lr"], color=color, linewidth=1.8)

        legend_patches.append(mpatches.Patch(color=color, label=label))

    # Baseline reference line on Top-1 panel
    ax_top1.axhline(
        BASELINE_TOP1, color="grey", linestyle="--", linewidth=1.2,
        label=f"Baseline (LR): {BASELINE_TOP1:.1%}",
    )

    # ── Labels & formatting ───────────────────────────────────────────────────
    ax_top1.set_title("Test Top-1 Accuracy");  ax_top1.set_ylabel("Accuracy")
    ax_top5.set_title("Test Top-5 Accuracy");  ax_top5.set_ylabel("Accuracy")
    ax_loss.set_title("Test Loss");             ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_lr.set_title("Learning Rate");           ax_lr.set_ylabel("LR");  ax_lr.set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("Epoch (global)")
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_xlim(left=0.5)

    # Format accuracy panels as percentages
    for ax in (ax_top1, ax_top5):
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    # Phase legend (shared background colours)
    phase_legend = [
        mpatches.Patch(color=PHASE_COLORS["phase1"], alpha=0.6, label="Phase 1 (head warmup)"),
        mpatches.Patch(color=PHASE_COLORS["phase2"], alpha=0.6, label="Phase 2 (end-to-end)"),
        mpatches.Patch(color="gold", label="Best epoch ★"),
    ]

    # Combine run labels + phase legend in the top-1 panel
    handles, labels_ = ax_top1.get_legend_handles_labels()
    ax_top1.legend(handles + phase_legend, labels_ + [p.get_label() for p in phase_legend],
                   fontsize=7.5, loc="lower right")

    # If multiple runs, add a separate legend for run colours
    if len(log_paths) > 1:
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=min(len(log_paths), 4), fontsize=8,
                   title="Runs", bbox_to_anchor=(0.5, -0.03))

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot finetune_log_*.csv training curves."
    )
    parser.add_argument(
        "logs", nargs="+", type=Path,
        help="Path(s) to finetune_log_*.csv files",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Save figure to this path instead of displaying (e.g. curves.png)",
    )
    args = parser.parse_args()

    missing = [p for p in args.logs if not p.exists()]
    if missing:
        print("ERROR — file(s) not found:", *missing, sep="\n  ", file=sys.stderr)
        sys.exit(1)

    plot_logs(args.logs, out_path=args.out)


if __name__ == "__main__":
    main()
