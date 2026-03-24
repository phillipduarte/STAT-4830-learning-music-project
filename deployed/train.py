import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_criterion(y_train_np: np.ndarray, n_classes: int,
                    label_smoothing: float, device: torch.device) -> nn.CrossEntropyLoss:
    """Inverse-frequency class-weighted cross-entropy loss."""
    counts  = np.bincount(y_train_np, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes  # mean weight ≈ 1
    class_weights = torch.from_numpy(weights).float().to(device)

    print(f"Class weight range: {weights.min():.3f} – {weights.max():.3f}")
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg):
    """Build LR scheduler from config. Supports 'cosine' and 'plateau'."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=cfg.scheduler_min_lr
        )
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}. Use 'cosine' or 'plateau'.")


# ── Core loop functions ───────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> None:
    """Single training epoch. Mutates model weights in place."""
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        criterion(model(X_batch), y_batch).backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> dict:
    """
    Evaluate model over a DataLoader (dropout disabled).

    Returns a dict with keys:
        loss, top1, top5, preds (tensor), labels (tensor)
    """
    model.eval()
    total_loss   = 0.0
    correct_top1 = 0
    correct_top5 = 0
    all_preds    = []
    all_labels   = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits  = model(X_batch)

        total_loss   += criterion(logits, y_batch).item() * len(y_batch)
        preds         = logits.argmax(dim=1)
        correct_top1 += preds.eq(y_batch).sum().item()
        correct_top5 += (
            logits.topk(5, dim=1).indices
            .eq(y_batch.unsqueeze(1)).any(dim=1).sum().item()
        )
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    n = len(loader.dataset)
    return {
        "loss":   total_loss / n,
        "top1":   correct_top1 / n,
        "top5":   correct_top5 / n,
        "preds":  torch.cat(all_preds),
        "labels": torch.cat(all_labels),
    }


# ── Full training run ─────────────────────────────────────────────────────────

def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
          cfg, criterion: nn.Module) -> dict:
    """
    Full training loop.

    Returns history dict with keys:
        train_loss, train_top1, train_top5,
        test_loss,  test_top1,  test_top5,  lr
    """
    device    = torch.device(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)

    history = {k: [] for k in
               ["train_loss", "train_top1", "train_top5",
                "test_loss",  "test_top1",  "test_top5", "lr"]}

    for epoch in range(1, cfg.n_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_m = evaluate(model, train_loader, criterion, device)
        test_m  = evaluate(model, test_loader,  criterion, device)

        # Plateau scheduler needs the metric; cosine just steps.
        if cfg.scheduler == "plateau":
            scheduler.step(test_m["top1"])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss", train_m["loss"]), ("train_top1", train_m["top1"]),
                     ("train_top5", train_m["top5"]), ("test_loss",  test_m["loss"]),
                     ("test_top1",  test_m["top1"]),  ("test_top5",  test_m["top5"]),
                     ("lr", current_lr)]:
            history[k].append(v)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{cfg.n_epochs} | "
                f"train loss: {train_m['loss']:.4f}  "
                f"train top-1: {train_m['top1']:.3f}  "
                f"test top-1: {test_m['top1']:.3f}  "
                f"test top-5: {test_m['top5']:.3f}  "
                f"lr: {current_lr:.2e}"
            )

    print(f"\nFinal test top-1: {history['test_top1'][-1]:.4f}")
    print(f"Final test top-5: {history['test_top5'][-1]:.4f}")
    return history