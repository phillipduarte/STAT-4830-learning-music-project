"""
train_mlp.py
Music Piece Classification — MLP on MERT Embeddings
Designed to run on Prime Intellect with a persistent storage volume.

Usage:
    python train_mlp.py --data-dir /mnt/storage/embeddings --out-dir /mnt/storage/outputs

Expects in --data-dir:
    embeddings_train.npy  (N_train, 768)
    embeddings_test.npy   (N_test,  768)
    labels_train.npy      (N_train,)  string labels
    labels_test.npy       (N_test,)   string labels
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on a remote GPU node
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MLP on MERT embeddings")
    p.add_argument("--data-dir",  type=Path, required=True,
                   help="Directory containing the .npy embedding files")
    p.add_argument("--out-dir",   type=Path, required=True,
                   help="Directory for plots, checkpoints, and logs")

    # Architecture
    p.add_argument("--hidden-dim",  type=int,   default=512)
    p.add_argument("--dropout",     type=float, default=0.5)

    # Optimizer
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-3)

    # Training
    p.add_argument("--epochs",      type=int,   default=1000)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--seed",        type=int,   default=42)

    # Evaluation display
    p.add_argument("--top-n-confused",      type=int, default=20)
    p.add_argument("--top-n-classes-chart", type=int, default=40)

    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    One-hidden-layer MLP: Linear(d, H) → ReLU → Dropout(p) → Linear(H, C)
    Returns raw logits; CrossEntropyLoss handles softmax.
    """
    def __init__(self, d: int, hidden_dim: int, n_classes: int, dropout_p: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct_top1, correct_top5 = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)

            preds = logits.argmax(dim=1)
            correct_top1 += preds.eq(y_batch).sum().item()

            top5 = logits.topk(5, dim=1).indices
            correct_top5 += top5.eq(y_batch.unsqueeze(1)).any(dim=1).sum().item()

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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, n_epochs, hidden_dim, lr, dropout_p, n_classes, out_dir):
    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(epochs, history["test_loss"],  label="Test",  color="coral", linestyle="--")
    axes[0].set(xlabel="Epoch", ylabel="CE Loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_top1"], label="Train top-1", color="steelblue")
    axes[1].plot(epochs, history["test_top1"],  label="Test top-1",  color="coral", linestyle="--")
    axes[1].plot(epochs, history["train_top5"], label="Train top-5", color="royalblue", alpha=0.5)
    axes[1].plot(epochs, history["test_top5"],  label="Test top-5",  color="tomato",    alpha=0.5, linestyle="--")
    axes[1].axhline(1/n_classes, color="gray", linestyle=":", label=f"Random ({1/n_classes:.3f})")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["lr"], color="mediumpurple")
    axes[2].set(xlabel="Epoch", ylabel="LR", title="LR Schedule")
    axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"MLP (768→{hidden_dim}→{n_classes}) | lr={lr} | dropout={dropout_p}", fontsize=10)
    plt.tight_layout()
    path = out_dir / "mlp_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_class(final, le, n_classes, top_n, out_dir):
    y_pred = final["preds"].numpy()
    y_true = final["labels"].numpy()

    per_class_acc = {}
    for i, name in enumerate(le.classes_):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[name] = (y_pred[mask] == i).mean()

    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    accs_all = list(per_class_acc.values())

    print(f"Median per-class acc: {np.median(accs_all):.4f}")
    print(f"Mean   per-class acc: {np.mean(accs_all):.4f}")
    print(f"Classes at 100%: {sum(1 for v in accs_all if v == 1.0)}")
    print(f"Classes at   0%: {sum(1 for v in accs_all if v == 0.0)}")

    n_show = min(top_n, len(sorted_classes))
    half   = n_show // 2
    names  = [c for c, _ in sorted_classes]
    accs   = [a for _, a in sorted_classes]
    dn = names[:half] + names[-half:]
    da = accs[:half]  + accs[-half:]
    colors = ["#4CAF50"] * half + ["#F44336"] * half

    fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.3)))
    ax.barh(dn, da, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(final["top1"], color="black", linestyle="--", linewidth=1.2,
               label=f"Overall ({final['top1']:.2f})")
    ax.set(xlabel="Per-Class Accuracy", title=f"Per-Class Accuracy — Top/Bottom {half}")
    ax.set_xlim(0, 1.05); ax.legend()
    ax.set_yticklabels([n.split("__")[-1] for n in dn], fontsize=8)
    plt.tight_layout()
    path = out_dir / "mlp_per_class_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return accs_all


def plot_confusion(final, le, n_classes, top_n, out_dir):
    y_pred = final["preds"].numpy()
    y_true = final["labels"].numpy()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    cm_err = cm.copy(); np.fill_diagonal(cm_err, 0)

    flat = np.argsort(cm_err.ravel())[::-1][:top_n]
    ti, pi = np.unravel_index(flat, cm_err.shape)
    involved = sorted(set(ti) | set(pi))
    sub_cm = cm[np.ix_(involved, involved)]
    names_short = [le.classes_[i].split("__")[-1] for i in involved]

    fig, ax = plt.subplots(figsize=(max(10, len(involved)*0.6), max(8, len(involved)*0.6)))
    sns.heatmap(sub_cm, xticklabels=names_short, yticklabels=names_short,
                annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix — {len(involved)} Most Confused Classes")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = out_dir / "mlp_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print("\nTop confused pairs:")
    for t, p_ in zip(ti, pi):
        c = cm[t, p_]
        if c > 0:
            print(f"  {le.classes_[t].split('__')[-1]:<30} → "
                  f"{le.classes_[p_].split('__')[-1]:<30} ({c})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    X_train = np.load(args.data_dir / "embeddings_train.npy")
    X_test  = np.load(args.data_dir / "embeddings_test.npy")
    y_train_str = np.load(args.data_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(args.data_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder()
    le.fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)

    n_classes = len(le.classes_)
    d = X_train.shape[1]
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  d={d}  C={n_classes}")

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(EmbeddingDataset(X_test, y_test),
                              batch_size=args.batch_size, shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MLP(d, args.hidden_dim, n_classes, args.dropout).to(device)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss: inverse-frequency class weighting + label smoothing ──────────
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).float().to(device),
        label_smoothing=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ──────────────────────────────────────────────────────
    history = {k: [] for k in
               ["train_loss","train_top1","train_top5","test_loss","test_top1","test_top5","lr"]}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        tr = evaluate(model, train_loader, criterion, device)
        te = evaluate(model, test_loader,  criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss", tr["loss"]), ("train_top1", tr["top1"]),
                     ("train_top5", tr["top5"]), ("test_loss",  te["loss"]),
                     ("test_top1",  te["top1"]), ("test_top5",  te["top5"]),
                     ("lr", lr)]:
            history[k].append(v)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{args.epochs} | "
                  f"train {tr['top1']:.3f} | test {te['top1']:.3f} | "
                  f"test-5 {te['top5']:.3f} | lr {lr:.2e}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.out_dir / "mlp_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "label_encoder_classes": le.classes_,
        "history": history,
        "final_top1": history["test_top1"][-1],
        "final_top5": history["test_top5"][-1],
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── Final evaluation & plots ───────────────────────────────────────────
    final = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test top-1: {final['top1']:.4f}  top-5: {final['top5']:.4f}")

    plot_training_curves(history, args.epochs, args.hidden_dim,
                         args.lr, args.dropout, n_classes, args.out_dir)
    accs_all = plot_per_class(final, le, n_classes, args.top_n_classes_chart, args.out_dir)
    plot_confusion(final, le, n_classes, args.top_n_confused, args.out_dir)

    print("\n" + "="*55)
    print(f"  Top-1: {final['top1']:.4f} ({final['top1']*100:.1f}%)")
    print(f"  Top-5: {final['top5']:.4f} ({final['top5']*100:.1f}%)")
    print(f"  Median per-class: {np.median(accs_all):.4f}")
    print(f"  Random baseline:  {1/n_classes:.4f} ({100/n_classes:.2f}%)")
    print("="*55)


if __name__ == "__main__":
    main()