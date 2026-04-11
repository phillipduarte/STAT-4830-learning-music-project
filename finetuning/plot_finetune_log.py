"""
plot_finetune_log.py — Visualise training logs produced by finetune_caching_disk.py.

Each run writes a CSV to data/finetune/finetune_log_<timestamp>.csv with columns:
    phase, epoch, test_top1, test_top5, test_loss, lr

Usage (single run):
    python plot_finetune_log.py data/finetune/finetune_log_20240101_120000.csv

Usage (compare multiple runs):
    python plot_finetune_log.py run1.csv run2.csv run3.csv

Options:
    --out PATH      Save figure to file instead of showing interactively.
                    Extension determines format (.png, .pdf, .svg, etc.)
    --dpi N         Resolution for raster outputs (default: 150)
    --no-phase-sep  Disable the vertical line separating Phase 1 / Phase 2

Examples:
    python plot_finetune_log.py data/finetune/finetune_log_*.csv --out comparison.png
    python plot_finetune_log.py my_log.csv --out training.pdf --dpi 300
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(path: Path) -> pd.DataFrame:
    """Load a finetune CSV log and add a monotonic global_step column."""
    df = pd.read_csv(path)

    required = {"phase", "epoch", "test_top1", "test_top5", "test_loss", "lr"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    # Build a monotonic step index across phase1 → phase2
    df = df.reset_index(drop=True)
    df["global_step"] = range(1, len(df) + 1)
    return df


def phase_boundaries(df: pd.DataFrame) -> list[float]:
    """
    Return the global_step values where the phase changes, for drawing
    vertical separator lines.
    """
    boundaries = []
    for i in range(1, len(df)):
        if df.loc[i, "phase"] != df.loc[i - 1, "phase"]:
            # Place the line midway between the two steps
            boundaries.append((df.loc[i, "global_step"] + df.loc[i - 1, "global_step"]) / 2)
    return boundaries


def label_for(path: Path, idx: int) -> str:
    """Short label: use the timestamp from the filename, or a generic name."""
    name = path.stem  # e.g. finetune_log_20240101_120000
    parts = name.split("_")
    # Try to extract the timestamp suffix (last two underscore-separated tokens)
    if len(parts) >= 2:
        suffix = "_".join(parts[-2:])
        if suffix.replace("_", "").isdigit():
            return suffix
    return f"run{idx + 1}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_logs(
    logs:          list[tuple[str, pd.DataFrame]],
    phase_sep:     bool = True,
    out:           Path | None = None,
    dpi:           int  = 150,
) -> None:
    """
    Four-panel figure:
        [0] Test Top-1 Accuracy
        [1] Test Top-5 Accuracy
        [2] Test Loss
        [3] Learning Rate
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    panel_cfg = [
        dict(col="test_top1", ylabel="Top-1 Accuracy",  title="Test Top-1 Accuracy",  pct=True),
        dict(col="test_top5", ylabel="Top-5 Accuracy",  title="Test Top-5 Accuracy",  pct=True),
        dict(col="test_loss", ylabel="Cross-Entropy",   title="Test Loss",             pct=False),
        dict(col="lr",        ylabel="Learning Rate",   title="Learning Rate",         pct=False, log_y=True),
    ]

    for i, (label, df) in enumerate(logs):
        color = COLORS[i % len(COLORS)]
        x     = df["global_step"]

        for ax, cfg in zip(axes, panel_cfg):
            ax.plot(
                x, df[cfg["col"]],
                color=color, linewidth=1.6, alpha=0.85,
                label=label,
            )

    # Draw phase-separator and phase-label annotations (use first log as reference)
    if phase_sep and logs:
        ref_df = logs[0][1]
        for bnd in phase_boundaries(ref_df):
            for ax in axes:
                ax.axvline(bnd, color="black", linewidth=1.0, linestyle="--", alpha=0.4)

        # Annotate phase bands using the first log
        phases = ref_df.groupby("phase", sort=False)["global_step"]
        for phase_name, steps in phases:
            mid = (steps.min() + steps.max()) / 2
            for ax in axes:
                ax.text(
                    mid, 1.01, phase_name.replace("phase", "Phase "),
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color="dimgray",
                )

    # Formatting
    for ax, cfg in zip(axes, panel_cfg):
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step (epoch)", fontsize=9)
        ax.set_ylabel(cfg["ylabel"],           fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        if cfg.get("pct"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        if cfg.get("log_y"):
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Add the baseline annotation to the top-1 panel
    axes[0].axhline(0.344, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    axes[0].text(
        0.01, 0.344 + 0.005, "Baseline 34.4%",
        transform=axes[0].get_yaxis_transform(),
        ha="left", va="bottom", fontsize=7.5, color="gray",
    )

    # Shared legend below the figure (only if multiple runs)
    if len(logs) > 1:
        handles = [
            Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=2, label=lbl)
            for i, (lbl, _) in enumerate(logs)
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=min(len(logs), 6), fontsize=9,
            frameon=True, bbox_to_anchor=(0.5, -0.04),
        )
    else:
        # Single run: legend inside the accuracy panel
        axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("Fine-tuning Training History", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot finetune_caching_disk.py training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "logs", nargs="+", type=Path,
        help="One or more finetune_log_*.csv files to plot.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output file path (e.g. training.png). Omit to show interactively.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for raster outputs (default: 150).",
    )
    parser.add_argument(
        "--no-phase-sep", action="store_true",
        help="Disable the vertical Phase 1 / Phase 2 separator line.",
    )
    args = parser.parse_args()

    loaded = []
    for idx, path in enumerate(args.logs):
        if not path.exists():
            print(f"ERROR: file not found — {path}", file=sys.stderr)
            sys.exit(1)
        try:
            df    = load_log(path)
            label = label_for(path, idx)
            loaded.append((label, df))
            print(f"Loaded {path.name}  ({len(df)} epochs, phases: {df['phase'].unique().tolist()})")
        except Exception as exc:
            print(f"ERROR loading {path.name}: {exc}", file=sys.stderr)
            sys.exit(1)

    plot_logs(
        logs      = loaded,
        phase_sep = not args.no_phase_sep,
        out       = args.out,
        dpi       = args.dpi,
    )


if __name__ == "__main__":
    main()