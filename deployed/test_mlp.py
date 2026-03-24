"""
test_mlp.py
Smoke-test for train_mlp.py — validates the full pipeline in ~60s.
Run this before committing to a full training job on Prime Intellect.

Usage:
    python test_mlp.py --data-dir /mnt/storage/embeddings --out-dir /mnt/storage/outputs

Checks (in order):
    1. Data files exist and shapes are sane
    2. Label encoder covers all test labels (no unseen classes)
    3. Model forward pass produces correct output shape
    4. Loss is finite and ~log(C) at init (sanity check for random weights)
    5. Backward pass runs without error; gradients are non-null
    6. 5-epoch training loop completes; loss decreases
    7. Checkpoint save/load round-trip
    8. Plot functions run without error (headless)
    9. evaluate() metrics are in valid ranges [0, 1]
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# ── Import from your training script ─────────────────────────────────────────
# Both files should live in the same directory.
try:
    from train_mlp import MLP, EmbeddingDataset, evaluate, plot_training_curves, plot_per_class, plot_confusion
except ImportError as e:
    print(f"[FAIL] Could not import from train_mlp.py: {e}")
    print("       Make sure test_mlp.py and train_mlp.py are in the same directory.")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def check(label: str, fn):
    """Run fn(), print PASS/FAIL, return True/False. Non-fatal by default."""
    try:
        result = fn()
        msg = f" — {result}" if isinstance(result, str) else ""
        print(f"{PASS} {label}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {label}")
        traceback.print_exc()
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test for train_mlp.py")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir",  type=Path, required=True)
    p.add_argument("--smoke-epochs", type=int, default=5,
                   help="Number of epochs to run in the training loop check (default: 5)")
    return p.parse_args()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_files(data_dir: Path):
    expected = [
        "embeddings_train.npy",
        "embeddings_test.npy",
        "labels_train.npy",
        "labels_test.npy",
    ]
    missing = [f for f in expected if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    return f"all 4 files found in {data_dir}"


def check_shapes(data_dir: Path):
    X_train = np.load(data_dir / "embeddings_train.npy")
    X_test  = np.load(data_dir / "embeddings_test.npy")
    y_train = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    y_test  = np.load(data_dir / "labels_test.npy",  allow_pickle=True)

    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
    assert X_test.ndim  == 2, f"X_test should be 2D, got {X_test.ndim}D"
    assert X_train.shape[1] == X_test.shape[1], "Embedding dims don't match"
    assert len(X_train) == len(y_train), "X_train / y_train length mismatch"
    assert len(X_test)  == len(y_test),  "X_test / y_test length mismatch"
    assert X_train.shape[1] == 768, f"Expected d=768 (MERT), got {X_train.shape[1]}"

    return (f"train={X_train.shape}, test={X_test.shape}, "
            f"d={X_train.shape[1]}, y_train={y_train.shape}, y_test={y_test.shape}")


def check_label_encoder(data_dir: Path):
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(data_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder()
    le.fit(y_train_str)

    unseen = set(y_test_str) - set(le.classes_)
    if unseen:
        raise ValueError(f"{len(unseen)} test label(s) unseen in train: {list(unseen)[:5]}")

    n_classes = len(le.classes_)
    return f"{n_classes} classes, 0 unseen test labels"


def check_forward_pass(data_dir: Path, device: torch.device):
    X_train = np.load(data_dir / "embeddings_train.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    le = LabelEncoder().fit(y_train_str)
    n_classes = len(le.classes_)
    d = X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    dummy = torch.randn(8, d).to(device)
    out = model(dummy)

    assert out.shape == (8, n_classes), f"Expected (8, {n_classes}), got {out.shape}"
    return f"output shape ({8}, {n_classes}) ✓"


def check_init_loss(data_dir: Path, device: torch.device):
    """
    At random init, CE loss should be ~log(C). A large deviation suggests a
    bug in the loss setup (e.g. wrong number of classes, broken class weights).
    """
    X_train = np.load(data_dir / "embeddings_train.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    le = LabelEncoder().fit(y_train_str)
    n_classes = len(le.classes_)
    d = X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.0).to(device)
    dummy_x = torch.randn(64, d).to(device)
    dummy_y = torch.randint(0, n_classes, (64,)).to(device)
    logits  = model(dummy_x)
    loss    = nn.CrossEntropyLoss()(logits, dummy_y)

    expected = np.log(n_classes)
    ratio = loss.item() / expected

    assert 0.5 < ratio < 2.0, (
        f"Init loss {loss.item():.3f} far from log(C)={expected:.3f} "
        f"(ratio={ratio:.2f}). Check model or loss setup."
    )
    return f"loss={loss.item():.3f}, log(C)={expected:.3f}, ratio={ratio:.2f}"


def check_backward(data_dir: Path, device: torch.device):
    X_train = np.load(data_dir / "embeddings_train.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    le = LabelEncoder().fit(y_train_str)
    n_classes = len(le.classes_)
    d = X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    opt = torch.optim.Adam(model.parameters())
    dummy_x = torch.randn(16, d).to(device)
    dummy_y = torch.randint(0, n_classes, (16,)).to(device)

    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(dummy_x), dummy_y)
    loss.backward()
    opt.step()

    null_grads = [n for n, p in model.named_parameters() if p.grad is None]
    if null_grads:
        raise RuntimeError(f"Null gradients on: {null_grads}")
    return "gradients non-null on all parameters"


def check_training_loop(data_dir: Path, out_dir: Path,
                         device: torch.device, n_epochs: int):
    """Run a real mini training loop and confirm loss decreases."""
    X_train = np.load(data_dir / "embeddings_train.npy")
    X_test  = np.load(data_dir / "embeddings_test.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(data_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder().fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)
    n_classes, d = len(le.classes_), X_train.shape[1]

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                              batch_size=256, shuffle=True)
    test_loader  = DataLoader(EmbeddingDataset(X_test, y_test),
                              batch_size=256, shuffle=False)

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).float().to(device),
        label_smoothing=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            criterion(model(X_b), y_b).backward()
            opt.step()
        m = evaluate(model, test_loader, criterion, device)
        losses.append(m["loss"])
        print(f"         epoch {epoch}/{n_epochs} | test loss {m['loss']:.4f} | top-1 {m['top1']:.3f}")

    # Loss should trend down over 5 epochs (allow some noise: just check last < first)
    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Loss did not decrease: epoch1={losses[0]:.4f}, "
            f"epoch{n_epochs}={losses[-1]:.4f}"
        )
    return (f"loss {losses[0]:.4f} → {losses[-1]:.4f} "
            f"(↓ {losses[0]-losses[-1]:.4f}) over {n_epochs} epochs")


def check_checkpoint(data_dir: Path, out_dir: Path, device: torch.device):
    X_train = np.load(data_dir / "embeddings_train.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    le = LabelEncoder().fit(y_train_str)
    n_classes, d = len(le.classes_), X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    path  = out_dir / "_smoke_test_ckpt.pt"
    torch.save({"model_state": model.state_dict()}, path)

    model2 = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    ckpt = torch.load(path, map_location=device)
    model2.load_state_dict(ckpt["model_state"])
    path.unlink()  # clean up

    dummy = torch.randn(4, d).to(device)
    model.eval(); model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(dummy), model2(dummy)), "Outputs differ after reload"
    return "save → load → inference outputs match"


def check_plots(data_dir: Path, out_dir: Path, device: torch.device):
    X_train = np.load(data_dir / "embeddings_train.npy")
    X_test  = np.load(data_dir / "embeddings_test.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(data_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder().fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)
    n_classes, d = len(le.classes_), X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.5).to(device)
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).float().to(device),
        label_smoothing=0.1,
    )
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test),
                             batch_size=256, shuffle=False)

    final = evaluate(model, test_loader, criterion, device)

    # Minimal history stub (3 epochs of fake data)
    stub_history = {k: [0.1, 0.09, 0.08] for k in
                    ["train_loss","test_loss","train_top1","test_top1",
                     "train_top5","test_top5","lr"]}

    smoke_dir = out_dir / "_smoke_plots"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(stub_history, 3, 512, 1e-3, 0.5, n_classes, smoke_dir)
    plot_per_class(final, le, n_classes, 20, smoke_dir)
    plot_confusion(final, le, n_classes, 10, smoke_dir)

    # Clean up smoke plot files
    for f in smoke_dir.iterdir():
        f.unlink()
    smoke_dir.rmdir()

    return "all 3 plot functions ran without error (headless)"


def check_metrics_range(data_dir: Path, device: torch.device):
    X_test  = np.load(data_dir / "embeddings_test.npy")
    X_train = np.load(data_dir / "embeddings_train.npy")
    y_train_str = np.load(data_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(data_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder().fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)
    n_classes, d = len(le.classes_), X_train.shape[1]

    model = MLP(d=d, hidden_dim=512, n_classes=n_classes, dropout_p=0.0).to(device)
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device))
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test),
                             batch_size=256, shuffle=False)

    m = evaluate(model, test_loader, criterion, device)
    assert 0.0 <= m["top1"] <= 1.0, f"top1={m['top1']} out of range"
    assert 0.0 <= m["top5"] <= 1.0, f"top5={m['top5']} out of range"
    assert m["top5"] >= m["top1"],  f"top5 < top1 ({m['top5']} < {m['top1']})"
    assert m["loss"]  > 0.0,        f"loss={m['loss']} should be > 0"
    return (f"top1={m['top1']:.3f}, top5={m['top5']:.3f}, "
            f"loss={m['loss']:.3f} — all in valid range")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.out_dir}")
    print("=" * 60)

    results = []

    print("\n── 1. File existence ────────────────────────────────────────")
    ok = check("Data files present", lambda: check_files(args.data_dir))
    results.append(ok)
    if not ok:
        print("\n[ABORT] Cannot proceed without data files.")
        sys.exit(1)

    print("\n── 2. Array shapes ──────────────────────────────────────────")
    results.append(check("Shapes and embedding dim",
                         lambda: check_shapes(args.data_dir)))

    print("\n── 3. Label encoder ─────────────────────────────────────────")
    results.append(check("No unseen test labels",
                         lambda: check_label_encoder(args.data_dir)))

    print("\n── 4. Forward pass ──────────────────────────────────────────")
    results.append(check("Forward pass output shape",
                         lambda: check_forward_pass(args.data_dir, device)))

    print("\n── 5. Init loss sanity ──────────────────────────────────────")
    results.append(check("Init loss ≈ log(C)",
                         lambda: check_init_loss(args.data_dir, device)))

    print("\n── 6. Backward pass ─────────────────────────────────────────")
    results.append(check("Backward pass + gradients",
                         lambda: check_backward(args.data_dir, device)))

    print(f"\n── 7. Training loop ({args.smoke_epochs} epochs) ──────────────────────")
    results.append(check(f"Loss decreases over {args.smoke_epochs} epochs",
                         lambda: check_training_loop(
                             args.data_dir, args.out_dir, device, args.smoke_epochs)))

    print("\n── 8. Checkpoint round-trip ─────────────────────────────────")
    results.append(check("Save → load → outputs match",
                         lambda: check_checkpoint(args.data_dir, args.out_dir, device)))

    print("\n── 9. Plot functions (headless) ─────────────────────────────")
    results.append(check("All plots render without error",
                         lambda: check_plots(args.data_dir, args.out_dir, device)))

    print("\n── 10. Metric ranges ─────────────────────────────────────────")
    results.append(check("top1/top5/loss in valid ranges",
                         lambda: check_metrics_range(args.data_dir, device)))

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 60)
    print(f"Results: {n_pass}/{len(results)} passed")
    if n_fail == 0:
        print("All checks passed — safe to launch a full training job.")
    else:
        print(f"{n_fail} check(s) failed — fix before launching.")
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()