import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineArcFaceModel(nn.Module):
    """
    Projection-based classifier over fixed embeddings with three head options:
      - softmax: linear classifier on projected embedding
      - cosine: scaled cosine similarity classifier
      - arcface: cosine classifier with additive angular margin for target class

    Forward returns class logits. For arcface training, pass labels to apply margin.
    """

    def __init__(
        self,
        d: int,
        n_classes: int,
        hidden_dim: int,
        embed_dim: int,
        dropout_p: float,
        activation: str = "gelu",
        head_type: str = "arcface",
        scale: float = 24.0,
        margin: float = 0.25,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        activation = activation.lower()
        if activation == "gelu":
            act_layer = nn.GELU
            nonlinearity = "relu"
        elif activation == "relu":
            act_layer = nn.ReLU
            nonlinearity = "relu"
        else:
            raise ValueError(f"Unsupported activation: {activation!r}. Use 'gelu' or 'relu'.")

        head_type = head_type.lower()
        if head_type not in {"softmax", "cosine", "arcface"}:
            raise ValueError(
                f"Unsupported head_type: {head_type!r}. Use 'softmax', 'cosine', or 'arcface'."
            )

        self.head_type = head_type
        self.scale = float(scale)
        self.margin = float(margin)

        norm_layer = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)

        self.projection = nn.Sequential(
            nn.Linear(d, hidden_dim),
            norm_layer,
            act_layer(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )

        if self.head_type == "softmax":
            self.classifier = nn.Linear(embed_dim, n_classes)
        else:
            self.class_weight = nn.Parameter(torch.empty(n_classes, embed_dim))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.head_type != "softmax":
            nn.init.xavier_uniform_(self.class_weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=1)

    def _cosine_logits(self, emb: torch.Tensor) -> torch.Tensor:
        normalized_w = F.normalize(self.class_weight, p=2, dim=1)
        cosine = F.linear(emb, normalized_w)
        return self.scale * cosine

    def _arcface_logits(self, emb: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        normalized_w = F.normalize(self.class_weight, p=2, dim=1)
        cosine = F.linear(emb, normalized_w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return self.scale * cosine

        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return self.scale * logits

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.encode(x)

        if self.head_type == "softmax":
            return self.classifier(emb)
        if self.head_type == "cosine":
            return self._cosine_logits(emb)
        return self._arcface_logits(emb, labels)
