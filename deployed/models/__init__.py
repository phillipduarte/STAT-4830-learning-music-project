from models.mlp import MLP
from models.cosine_arcface import CosineArcFaceModel

# Maps the model_name string in any config to its class.
# To add a new architecture: implement it in models/<name>.py,
# import it here, and add one line to this dict.
REGISTRY: dict[str, type] = {
    "mlp": MLP,
    "cosine_arcface": CosineArcFaceModel,
}


def build_model(cfg, d: int, n_classes: int):
    """
    Instantiate a model from the registry using the provided config.

    Each model class is responsible for declaring which kwargs it needs;
    this function passes only the fields that exist on the config object
    and are accepted by the model's __init__. Adding a new architecture
    never requires changes here — only a new entry in REGISTRY and a new
    config dataclass with the right field names.

    Args:
        cfg:       A config dataclass (subclass of BaseConfig).
        d:         Input embedding dimension.
        n_classes: Number of output classes.

    Returns:
        An nn.Module on cfg.device.
    """
    import inspect
    import torch

    model_cls = REGISTRY[cfg.model_name]

    # Collect kwargs accepted by the model's __init__ (excluding self, d, n_classes).
    sig = inspect.signature(model_cls.__init__)
    accepted = set(sig.parameters) - {"self", "d", "n_classes"}

    # Pull matching fields from the config dataclass.
    kwargs = {k: getattr(cfg, k) for k in accepted if hasattr(cfg, k)}

    model = model_cls(d=d, n_classes=n_classes, **kwargs)
    return model.to(torch.device(cfg.device))
