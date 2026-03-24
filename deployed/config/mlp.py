from dataclasses import dataclass
from config.base import BaseConfig


@dataclass
class MLPConfig(BaseConfig):
    # Identifies which model class to look up in the registry.
    model_name: str = "mlp"

    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_dim: int = 512
    dropout_p: float = 0.5

    # ── Human-readable label for output files / plots ─────────────────────────
    @property
    def run_name(self) -> str:
        return (
            f"mlp_h{self.hidden_dim}_do{self.dropout_p}"
            f"_lr{self.lr}_wd{self.weight_decay}"
        )