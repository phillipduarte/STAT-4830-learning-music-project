import numpy as np
import inspect
import copy
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


def build_optimizer(model: nn.Module, cfg):
    """Build optimizer from config. Supports 'adam' and 'adamw'."""
    opt = cfg.optimizer.lower()
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}. Use 'adam' or 'adamw'.")


# ── Core loop functions ───────────────────────────────────────────────────────

def _forward_logits(model: nn.Module, X_batch: torch.Tensor,
                    y_batch: torch.Tensor | None = None) -> torch.Tensor:
    """Call model forward with labels when supported by the model signature."""
    params = inspect.signature(model.forward).parameters
    if "labels" in params:
        return model(X_batch, labels=y_batch)
    if len(params) >= 2:
        return model(X_batch, y_batch)
    return model(X_batch)


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device) -> None:
    """Single training epoch. Mutates model weights in place."""
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = _forward_logits(model, X_batch, y_batch)
        criterion(logits, y_batch).backward()
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
        logits  = _forward_logits(model, X_batch)

        total_loss   += criterion(logits, y_batch).item() * len(y_batch)
        preds         = logits.argmax(dim=1)
        correct_top1 += preds.eq(y_batch).sum().item()
        k = min(5, logits.shape[1])
        correct_top5 += (
            logits.topk(k, dim=1).indices
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

def train(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    cfg,
    criterion: nn.Module,
    monitor_name: str = "val",
) -> dict:
    """
    Full training loop.

    Early stopping monitors validation top-1 when eval_loader is provided,
    otherwise it falls back to train top-1.

    Returns dict with keys:
        history, best_epoch, best_metric
    """
    device    = torch.device(cfg.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    metric_prefix = monitor_name if eval_loader is not None else "train"
    history = {k: [] for k in
               ["train_loss", "train_top1", "train_top5",
                f"{metric_prefix}_loss", f"{metric_prefix}_top1", f"{metric_prefix}_top5", "lr"]}

    best_metric = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    min_delta = float(cfg.early_stopping_min_delta)

    for epoch in range(1, cfg.n_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_m = evaluate(model, train_loader, criterion, device)
        eval_m = evaluate(model, eval_loader, criterion, device) if eval_loader is not None else train_m
        monitor_metric = eval_m["top1"]

        # Plateau scheduler needs the metric; cosine just steps.
        if cfg.scheduler == "plateau":
            scheduler.step(monitor_metric)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss", train_m["loss"]), ("train_top1", train_m["top1"]),
                     ("train_top5", train_m["top5"]), (f"{metric_prefix}_loss", eval_m["loss"]),
                     (f"{metric_prefix}_top1",  eval_m["top1"]),  (f"{metric_prefix}_top5",  eval_m["top5"]),
                     ("lr", current_lr)]:
            history[k].append(v)

        if monitor_metric > (best_metric + min_delta):
            best_metric = monitor_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{cfg.n_epochs} | "
                f"train loss: {train_m['loss']:.4f}  "
                f"train top-1: {train_m['top1']:.3f}  "
                f"{metric_prefix} top-1: {eval_m['top1']:.3f}  "
                f"{metric_prefix} top-5: {eval_m['top5']:.3f}  "
                f"lr: {current_lr:.2e}"
            )

        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} (best {metric_prefix}_top1={best_metric:.4f} "
                f"at epoch {best_epoch})"
            )
            break

    if cfg.save_best_only:
        model.load_state_dict(best_state)

    print(f"\nBest {metric_prefix} top-1: {best_metric:.4f} (epoch {best_epoch})")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
    }
