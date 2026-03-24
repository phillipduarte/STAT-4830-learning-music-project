from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class EmbeddingDataset(Dataset):
    """Thin wrapper around pre-computed embedding arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(embeddings_dir: Path, batch_size: int):
    """
    Load pre-computed MERT embeddings from disk and return DataLoaders
    along with metadata needed to build models and loss functions.

    Returns:
        train_loader: DataLoader
        test_loader:  DataLoader
        le:           Fitted LabelEncoder (use le.classes_ for class names)
        d:            Embedding dimension
        n_classes:    Number of classes
        y_train_np:   Raw integer labels for the training set (for class weighting)
    """
    X_train = np.load(embeddings_dir / "embeddings_train.npy", allow_pickle=True)
    X_test  = np.load(embeddings_dir / "embeddings_test.npy", allow_pickle=True)

    y_train_str = np.load(embeddings_dir / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(embeddings_dir / "labels_test.npy",  allow_pickle=True)

    le = LabelEncoder()
    le.fit(y_train_str)
    y_train = le.transform(y_train_str).astype(np.int64)
    y_test  = le.transform(y_test_str).astype(np.int64)

    train_loader = DataLoader(
        EmbeddingDataset(X_train, y_train),
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

    print(f"Train: {len(X_train):,} snippets  |  Test: {len(X_test):,} snippets")
    print(f"Embedding dim: {d}  |  Classes: {n_classes}")
    print(f"Random baseline: {1/n_classes:.4f} ({100/n_classes:.2f}%)")

    return train_loader, test_loader, le, d, n_classes, y_train