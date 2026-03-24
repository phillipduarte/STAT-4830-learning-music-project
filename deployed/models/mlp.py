import torch.nn as nn


class MLP(nn.Module):
    """
    One-hidden-layer MLP for multi-class classification over MERT embeddings.

    Architecture:
        Linear(d, H) → ReLU → Dropout(p) → Linear(H, C)

    Returns raw logits; loss function handles softmax.
    """

    def __init__(self, d: int, n_classes: int, hidden_dim: int, dropout_p: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, n_classes),
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)