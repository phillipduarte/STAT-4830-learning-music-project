from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class EmbeddingDataset(Dataset):
    """Thin wrapper around pre-computed embedding arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(
    embeddings_dir: Path,
    batch_size: int,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    include_val_in_train: bool = False,
):
    """
    Load pre-computed MERT embeddings from disk and return DataLoaders
    along with metadata needed to build models and loss functions.

    Returns:
        train_loader: DataLoader over train (or train+val when include_val_in_train=True)
        val_loader:   DataLoader or None
        test_loader:  DataLoader
        le:           Fitted LabelEncoder (use le.classes_ for class names)
        d:            Embedding dimension
        n_classes:    Number of classes
        y_train_np:   Raw integer labels used for the training loss weights
    """
    X_train = np.load(embeddings_dir / "embeddings_train.npy", allow_pickle=True)
    X_test  = np.load(embeddings_dir / "embeddings_test.npy", allow_pickle=True)

    y_train_str = np.load(embeddings_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(embeddings_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder()
    le.fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)

    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    if val_ratio > 0.0:
        all_idx = np.arange(len(y_train))
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=val_ratio,
            random_state=split_seed,
            stratify=y_train,
        )
        if np.intersect1d(train_idx, val_idx).size != 0:
            raise RuntimeError("Train/val split overlap detected.")

        X_train_split, y_train_split = X_train[train_idx], y_train[train_idx]
        X_val_split, y_val_split = X_train[val_idx], y_train[val_idx]
    else:
        X_train_split, y_train_split = X_train, y_train
        X_val_split, y_val_split = None, None

    if include_val_in_train and X_val_split is not None:
        X_train_used = np.concatenate([X_train_split, X_val_split], axis=0)
        y_train_used = np.concatenate([y_train_split, y_val_split], axis=0)
        val_loader = None
    else:
        X_train_used, y_train_used = X_train_split, y_train_split
        val_loader = (
            DataLoader(
                EmbeddingDataset(X_val_split, y_val_split),
                batch_size=batch_size,
                shuffle=False,
            )
            if X_val_split is not None
            else None
        )

    train_loader = DataLoader(
        EmbeddingDataset(X_train_used, y_train_used),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        EmbeddingDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    d         = X_train.shape[1]
    n_classes = len(le.classes_)

    if val_loader is None:
        print(
            f"Train: {len(X_train_used):,} snippets  |  "
            f"Val: none  |  Test: {len(X_test):,} snippets"
        )
    else:
        print(
            f"Train: {len(X_train_used):,} snippets  |  "
            f"Val: {len(X_val_split):,} snippets  |  Test: {len(X_test):,} snippets"
        )
    print(f"Embedding dim: {d}  |  Classes: {n_classes}")
    print(f"Random baseline: {1/n_classes:.4f} ({100/n_classes:.2f}%)")

    return train_loader, val_loader, test_loader, le, d, n_classes, y_train_used
