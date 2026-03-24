from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class BaseConfig:
    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Override this if your embeddings live elsewhere.
    embeddings_dir: Path = field(default_factory=lambda: Path("embeddings"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs: int = 1000
    batch_size: int = 256

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-3

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # "cosine" | "plateau"
    scheduler: str = "cosine"
    scheduler_patience: int = 50      # only used when scheduler="plateau"
    scheduler_factor: float = 0.5     # only used when scheduler="plateau"
    scheduler_min_lr: float = 1e-5

    # ── Loss ──────────────────────────────────────────────────────────────────
    label_smoothing: float = 0.1

    # ── Evaluation ────────────────────────────────────────────────────────────
    top_n_confused: int = 20
    top_n_classes_chart: int = 40

    def __post_init__(self):
        self.embeddings_dir = Path(self.embeddings_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)