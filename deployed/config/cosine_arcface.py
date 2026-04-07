from dataclasses import dataclass

from config.base import BaseConfig


@dataclass
class CosineArcFaceConfig(BaseConfig):
    # Identifies which model class to look up in the registry.
    model_name: str = "cosine_arcface"

    # Projection network.
    hidden_dim: int = 512
    embed_dim: int = 128
    dropout_p: float = 0.2
    activation: str = "gelu"       # "gelu" | "relu"
    use_layer_norm: bool = True

    # Classifier head.
    head_type: str = "arcface"     # "softmax" | "cosine" | "arcface"
    scale: float = 24.0
    margin: float = 0.25            # only used for arcface

    @property
    def run_name(self) -> str:
        return (
            f"{self.model_name}_{self.head_type}"
            f"_h{self.hidden_dim}_e{self.embed_dim}"
            f"_do{self.dropout_p}_act{self.activation}"
            f"_s{self.scale}_m{self.margin}"
            f"_lr{self.lr}_wd{self.weight_decay}"
        )
    # Optimizer defaults for this model family.
    optimizer: str = "adamw"
