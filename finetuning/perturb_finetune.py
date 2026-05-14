"""
perturb_finetune_v2.py — Improved fine-tuning of MERT + MLP on perturbed dataset.

Changes from perturb_finetune_caching.py:
  1. --unfreeze-layers default 2 → 6
  2. --phase2-scheduler cosine|plateau (default: cosine)
  3. --phase2-epochs default 30 → 60
  4. --hidden-dims list (default: 2048 512)
  5. --variants-per-snippet K + --aux-weight λ: paired variant sampler + cosine aux loss
       A VariantBatchSampler guarantees K variants of the same (piece, snippet_index)
       appear together in every batch. A cosine similarity auxiliary loss then
       explicitly pushes all K embeddings of the same snippet toward each other,
       directly encoding pitch/tempo invariance as a training objective.
       The total loss is: L = CrossEntropy + λ * (1 - cosine_similarity).
  6. --phase2-batch default 64 → 128

All changes are exposed as CLI args so nothing is hardcoded.

Usage:
  # Recommended
  python perturb_finetune_v2.py \\
    --unfreeze-layers 6 --phase2-epochs 60 --phase2-scheduler cosine \\
    --hidden-dims 2048 512 --variants-per-snippet 4 --aux-weight 0.3

  # No aux loss (variants-per-snippet=1 disables pairing)
  python perturb_finetune_v2.py --variants-per-snippet 1 --aux-weight 0.0

  # Reproduce original behaviour
  python perturb_finetune_v2.py \\
    --unfreeze-layers 2 --phase2-epochs 30 --phase2-scheduler plateau \\
    --hidden-dims 512 --variants-per-snippet 1 --aux-weight 0.0 --phase2-batch 64

Outputs (saved to perturb/finetune/):
  finetune_best_{timestamp}.pt    — best checkpoint
  finetune_log_{timestamp}.csv    — epoch-level training history
"""

import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SNIPPETS_DIR,
    AUDIO_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    SAMPLE_RATE,
    PERTURB_DIR
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# FINETUNE_DIR  = EMBEDDINGS_DIR.parent / "finetune"
# MANIFEST_PATH = SNIPPETS_DIR / "manifest.csv"

AUDIO_DIR = PERTURB_DIR / "psnippets_new"
EMBEDDINGS_DIR = PERTURB_DIR / "embeddings_new"
FINETUNE_DIR = PERTURB_DIR / "finetune"
MANIFEST_PATH = PERTURB_DIR / "manifest.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Phase 1 — head warmup on precomputed embeddings (fast)
PHASE1_EPOCHS   = 20
PHASE1_LR       = 1e-3
PHASE1_PATIENCE = 10
PHASE1_FACTOR   = 0.5
PHASE1_MIN_LR   = 1e-5
PHASE1_BATCH    = 256     # large is fine — no MERT forward pass

# Phase 2 — end-to-end fine-tuning with caching
PHASE2_EPOCHS   = 60
PHASE2_LR_HEAD  = 1e-3
PHASE2_LR_MERT  = 1e-5    # 100x lower — prevents catastrophic forgetting
PHASE2_PATIENCE = 10
PHASE2_FACTOR   = 0.5
PHASE2_MIN_LR   = 1e-6
PHASE2_BATCH    = 128     # increased from 64 to support paired variant sampling
WEIGHT_DECAY    = 1e-3

UNFREEZE_LAYERS      = 6
HIDDEN_DIMS          = [2048, 512]
VARIANTS_PER_SNIPPET = 4      # K variants per snippet guaranteed per batch
AUX_WEIGHT           = 0.3    # λ: weight of cosine similarity aux loss
LOG_EVERY       = 1
SEED            = 42


# ---------------------------------------------------------------------------
# Phase 1 — precomputed embedding datasets
# ---------------------------------------------------------------------------

def load_embedding_datasets(le: LabelEncoder) -> tuple[TensorDataset, TensorDataset]:
    """Load precomputed .npy embeddings into TensorDatasets for Phase 1."""
    X_train = np.load(EMBEDDINGS_DIR / "embeddings_train.npy")
    X_test  = np.load(EMBEDDINGS_DIR / "embeddings_test.npy")
    y_train = le.transform(np.load(EMBEDDINGS_DIR / "labels_train.npy", allow_pickle=True))
    y_test  = le.transform(np.load(EMBEDDINGS_DIR / "labels_test.npy",  allow_pickle=True))

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test.astype(np.int64)),
    )
    log.info(f"Precomputed embeddings loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Phase 2 — audio loading
# ---------------------------------------------------------------------------

def _load_wav(wav_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        import soundfile as sf
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        return waveform
    except Exception as e:
        log.warning(f"Failed to load {wav_path.name}: {e}. Using silence.")
        return np.zeros(target_sr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase 2 — frozen layer cache
# ---------------------------------------------------------------------------

def build_frozen_cache(
    mert,
    processor,
    rows: list[dict],
    audio_dir: Path,
    device: str,
    n_unfreeze: int,
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute and cache the output of the last frozen MERT layer for every snippet.

    With 12 layers and n_unfreeze=2, we cache hidden_states[10] — the output
    after layer 9, which is the input to the first unfrozen layer (layer 10).

    Resumable: if cache_path is provided, progress is checkpointed every 500
    snippets. If the process is interrupted and restarted, already-processed
    snippets are loaded from the checkpoint and the build resumes from where
    it left off. Pass force_rebuild=True to ignore any existing checkpoint.

    Returns:
        cache: dict mapping filename → (hidden_state, attention_mask)
            hidden_state:   float32 CPU tensor of shape (T_i, 768)
            attention_mask: bool CPU tensor of shape (T_i,), True = valid frame
    """
def _save_cache_checkpoint(cache: dict, cache_path: Path) -> None:
    """
    Save cache atomically: write to a temp file then rename.
    This prevents a mid-write disconnect from corrupting the checkpoint.
    Also keeps a .bak of the previous checkpoint so a corrupt latest
    can be recovered from.
    """
    tmp_path = cache_path.with_suffix(".tmp")
    bak_path = cache_path.with_suffix(".bak")
    torch.save(cache, tmp_path)
    if cache_path.exists():
        cache_path.rename(bak_path)   # rotate current → backup
    tmp_path.rename(cache_path)       # atomic replace


def _load_cache_checkpoint(cache_path: Path) -> dict:
    """
    Load cache checkpoint, falling back to .bak if the main file is corrupt.
    """
    for path in [cache_path, cache_path.with_suffix(".bak")]:
        if not path.exists():
            continue
        try:
            cache = torch.load(path, map_location="cpu")
            if path != cache_path:
                log.warning(f"Main checkpoint corrupt — loaded from backup: {path.name}")
            return cache
        except Exception as e:
            log.warning(f"Failed to load {path.name}: {e}")
    return {}


def build_frozen_cache(
    mert,
    processor,
    rows: list[dict],
    audio_dir: Path,
    device: str,
    n_unfreeze: int,
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute and cache the output of the last frozen MERT layer for every snippet.
    Resumable: checkpointed every 500 snippets using atomic writes.
    Falls back to .bak if the main checkpoint is corrupt.
    """
    CHECKPOINT_EVERY = 500

    n_total_layers  = len(mert.encoder.layers)
    cache_layer_idx = n_total_layers - n_unfreeze
    log.info(
        f"Building frozen cache: running layers 0–{cache_layer_idx - 1} "
        f"(top {n_unfreeze} unfrozen layers will run at train time)"
    )

    # Load existing checkpoint if present
    cache = {}
    if cache_path is not None and not force_rebuild:
        cache = _load_cache_checkpoint(cache_path)
        if cache:
            log.info(f"Resumed with {len(cache)} / {len(rows)} snippets already done.")

    # Filter to only rows not yet in cache
    remaining = [r for r in rows if r["filename"] not in cache]
    if not remaining:
        log.info("Cache already complete — nothing to build.")
        return cache

    log.info(f"Processing {len(remaining)} remaining snippets...")
    mert.eval()

    with torch.no_grad():
        for i, row in enumerate(remaining):
            wav_name = Path(row["filename"]).with_suffix(".wav").name
            waveform = _load_wav(audio_dir / wav_name)

            inputs = processor(
                [waveform],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            outputs = mert(**inputs, output_hidden_states=True)

            hidden    = outputs.hidden_states[cache_layer_idx].squeeze(0).cpu()
            attn_mask = torch.ones(hidden.shape[0], dtype=torch.bool)
            cache[row["filename"]] = (hidden, attn_mask)

            processed = i + 1
            if processed % 200 == 0 or processed == len(remaining):
                log.info(f"  Cache: {len(cache)}/{len(rows)} snippets total")

            # Checkpoint periodically so a disconnect doesn't lose everything
            if cache_path is not None and processed % CHECKPOINT_EVERY == 0:
                _save_cache_checkpoint(cache, cache_path)
                log.info(f"  Checkpoint saved ({len(cache)} snippets) → {cache_path.name}")

    # Save final complete cache
    if cache_path is not None:
        _save_cache_checkpoint(cache, cache_path)
        log.info(f"Cache complete and saved: {cache_path.name}")

    log.info(f"Cache built: {len(cache)} snippets, "
             f"~{sum(h.numel() * 4 for h, _ in cache.values()) / 1e9:.2f} GB in RAM")
    return cache


class CachedDataset(Dataset):
    """
    Dataset that returns pre-cached frozen-layer hidden states instead of raw audio.
    Each __getitem__ returns (hidden_state, attention_mask, label, snippet_key).
    snippet_key = (piece_id, snippet_index) — used by VariantBatchSampler.
    """

    def __init__(
        self,
        rows: list[dict],
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        label_encoder: LabelEncoder,
    ):
        self.rows  = rows
        self.cache = cache
        self.le    = label_encoder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        hidden, attn_mask = self.cache[row["filename"]]
        label = int(self.le.transform([row["label"]])[0])
        snippet_key = (row["piece_id"], row["snippet_index"])
        return hidden, attn_mask, label, snippet_key


class VariantBatchSampler:
    """
    Batch sampler that guarantees exactly K variants of each snippet appear
    together in the same batch.

    Strategy per batch:
      1. Sample (batch_size // K) distinct snippet keys at random.
      2. For each key, sample K of its variant row indices (with replacement
         if fewer than K variants exist for that key).
      3. Shuffle within the batch so variants aren't contiguous.

    The collate function uses the snippet_key field to identify pairs.

    Args:
        dataset:    CachedDataset — needs rows with piece_id and snippet_index.
        batch_size: Total items per batch. Should be divisible by K.
        K:          Number of variants per snippet per batch.
        shuffle:    Shuffle the order of snippet groups each epoch.
    """

    def __init__(self, dataset: CachedDataset, batch_size: int, K: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.K          = K
        self.shuffle    = shuffle

        # Group row indices by (piece_id, snippet_index)
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, row in enumerate(dataset.rows):
            groups[(row["piece_id"], row["snippet_index"])].append(idx)
        self.groups     = list(groups.values())   # list of lists of indices
        self.n_per_batch = batch_size // K        # number of distinct snippets per batch

    def __iter__(self):
        groups = [list(g) for g in self.groups]
        if self.shuffle:
            random.shuffle(groups)

        batch = []
        for group in groups:
            # Sample K indices from this group (with replacement if needed)
            sampled = random.choices(group, k=self.K) if len(group) < self.K \
                      else random.sample(group, k=self.K)
            batch.extend(sampled)

            if len(batch) >= self.batch_size:
                random.shuffle(batch)
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        # Yield final partial batch if non-empty
        if batch:
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return (len(self.groups) * self.K + self.batch_size - 1) // self.batch_size


def cached_collate_fn(batch):
    """
    Collate variable-length cached hidden states into a padded batch.
    Returns (padded_hidden, padded_mask, labels, snippet_keys).
    snippet_keys is a list of (piece_id, snippet_index) tuples — used to
    identify variant pairs for the auxiliary cosine loss.
    """
    hiddens, masks, labels, snippet_keys = zip(*batch)
    max_T = max(h.shape[0] for h in hiddens)
    D     = hiddens[0].shape[1]

    padded_hidden = torch.zeros(len(hiddens), max_T, D)
    padded_mask   = torch.zeros(len(hiddens), max_T, dtype=torch.bool)

    for i, (h, m) in enumerate(zip(hiddens, masks)):
        T = h.shape[0]
        padded_hidden[i, :T] = h
        padded_mask[i,   :T] = m

    return padded_hidden, padded_mask, torch.tensor(labels, dtype=torch.long), list(snippet_keys)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Variable-depth MLP classifier.
    hidden_dims: list of hidden layer sizes, e.g. [512], [2048], or [2048, 512].
    Each hidden layer: Linear → BN → ReLU → Dropout.
    """

    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout_p: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MERTClassifier(nn.Module):
    """
    MERT backbone + MLP head.

    Supports two forward modes:
      forward(inputs)                     — full pass through all MERT layers (cache build)
      forward_from_cache(hidden, mask)    — top N unfrozen layers only (Phase 2 training)
    """

    def __init__(self, mert_model, mlp_head: MLP, n_unfreeze: int):
        super().__init__()
        self.mert      = mert_model
        self.head      = mlp_head
        self.n_unfreeze = n_unfreeze

    def forward(self, inputs: dict) -> torch.Tensor:
        """Full forward pass — used for cache building only."""
        outputs = self.mert(**inputs, output_hidden_states=False)
        pooled  = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)

    def forward_from_cache(
        self,
        cached_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass from cache — returns logits."""
        return self.head(self.get_embeddings(cached_hidden, attention_mask))

    def get_embeddings(
        self,
        cached_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run top N unfrozen MERT layers and return pooled embeddings (before head).
        Used to compute the auxiliary cosine loss on the representation directly.
        """
        hidden = cached_hidden

        # Convert bool mask to additive attention mask: (B, 1, 1, T)
        # 0.0 for valid positions, -1e4 for padding (added to attention weights pre-softmax)
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e4
        ext_mask = ext_mask.to(hidden.device)

        # Run only the unfrozen top layers
        for layer in self.mert.encoder.layers[-self.n_unfreeze:]:
            layer_out = layer(hidden, attention_mask=ext_mask)
            hidden    = layer_out[0]

        # Final layer norm (if present — MERT has one after all encoder layers)
        if hasattr(self.mert.encoder, "layer_norm"):
            hidden = self.mert.encoder.layer_norm(hidden)

        # Masked mean pool: average over valid frames only
        mask_f = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return pooled


def freeze_mert(model: MERTClassifier) -> None:
    for param in model.mert.parameters():
        param.requires_grad = False
    log.info("MERT fully frozen.")


def unfreeze_top_layers(model: MERTClassifier, n_layers: int) -> None:
    encoder = model.mert.encoder
    for layer in encoder.layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    if hasattr(encoder, "layer_norm"):
        for param in encoder.layer_norm.parameters():
            param.requires_grad = True
    unfrozen = sum(p.numel() for p in model.mert.parameters() if p.requires_grad)
    total    = sum(p.numel() for p in model.mert.parameters())
    log.info(f"Unfroze top {n_layers} MERT layers: {unfrozen:,} / {total:,} MERT params trainable.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_embeddings(head: MLP, loader: DataLoader, criterion, device: str) -> dict:
    """Phase 1 eval — head only on precomputed embeddings."""
    head.eval()
    total_loss, correct_top1, correct_top5, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = head(X_batch)
            total_loss   += criterion(logits, y_batch).item() * len(y_batch)
            correct_top1 += logits.argmax(1).eq(y_batch).sum().item()
            top5          = logits.topk(min(5, logits.size(1)), dim=1).indices
            correct_top5 += top5.eq(y_batch.unsqueeze(1)).any(1).sum().item()
            n += len(y_batch)
    return {"loss": total_loss / n, "top1": correct_top1 / n, "top5": correct_top5 / n}


def evaluate_cached(model: MERTClassifier, loader: DataLoader, criterion, device: str) -> dict:
    """Phase 2 eval — top layers only, using cached hidden states."""
    model.eval()
    total_loss, correct_top1, correct_top5, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for hidden, mask, labels, _keys in loader:
            hidden = hidden.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)
            logits = model.forward_from_cache(hidden, mask)
            total_loss   += criterion(logits, labels).item() * len(labels)
            correct_top1 += logits.argmax(1).eq(labels).sum().item()
            top5          = logits.topk(min(5, logits.size(1)), dim=1).indices
            correct_top5 += top5.eq(labels.unsqueeze(1)).any(1).sum().item()
            n += len(labels)
    return {"loss": total_loss / n, "top1": correct_top1 / n, "top5": correct_top5 / n}


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def run_phase(
    phase_name:   str,
    train_fn,
    eval_fn,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    optimizer,
    scheduler,
    criterion,
    n_epochs:     int,
    device:       str,
    log_writer,
    save_fn,
    best_top1:    float = 0.0,
    log_every:    int   = LOG_EVERY,
) -> tuple[float, float]:
    """Generic training loop. Returns (best_top1, best_top5)."""
    best_top5 = 0.0

    for epoch in range(1, n_epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            train_fn(batch)
            if (batch_idx + 1) % 20 == 0:
                log.info(
                    f"  [{phase_name}] epoch {epoch}/{n_epochs} "
                    f"— batch {batch_idx+1}/{len(train_loader)}"
                )

        metrics    = eval_fn(test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        # ReduceLROnPlateau needs a metric; CosineAnnealingLR takes no argument
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics["top1"])
        else:
            scheduler.step()

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_top5 = metrics["top5"]
            save_fn({"top1": best_top1, "top5": best_top5, "phase": phase_name, "epoch": epoch})

        if epoch % log_every == 0 or epoch == n_epochs:
            log.info(
                f"[{phase_name}] epoch {epoch:>3}/{n_epochs}  "
                f"test top-1: {metrics['top1']:.4f}  "
                f"top-5: {metrics['top5']:.4f}  "
                f"loss: {metrics['loss']:.4f}  "
                f"lr: {current_lr:.2e}  "
                f"(best: {best_top1:.4f})"
            )
        log_writer.writerow({
            "phase": phase_name, "epoch": epoch,
            "test_top1": metrics["top1"], "test_top5": metrics["top5"],
            "test_loss": metrics["loss"], "lr": current_lr,
        })

    return best_top1, best_top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    phase1_epochs:        int   = PHASE1_EPOCHS,
    phase2_epochs:        int   = PHASE2_EPOCHS,
    unfreeze_layers:      int   = UNFREEZE_LAYERS,
    hidden_dims:          list  = HIDDEN_DIMS,
    phase2_scheduler:     str   = "cosine",
    variants_per_snippet: int   = VARIANTS_PER_SNIPPET,
    aux_weight:           float = AUX_WEIGHT,
    phase2_batch:         int   = PHASE2_BATCH,
    force_rebuild_cache:  bool  = False,
    device:               str   = EMBEDDING_DEVICE,
    model_id:             str   = EMBEDDING_MODEL,
) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FINETUNE_DIR / f"finetune_best_{run_id}.pt"
    log_path  = FINETUNE_DIR / f"finetune_log_{run_id}.csv"
    log.info(f"Run ID: {run_id}")

    # Label encoder — matches existing pipeline label→index mapping
    y_train_str = np.load(EMBEDDINGS_DIR / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(EMBEDDINGS_DIR / "labels_test.npy",  allow_pickle=True)
    le = LabelEncoder()
    le.fit(y_train_str)
    n_classes = len(le.classes_)
    log.info(f"Classes: {n_classes}  |  Device: {device}")

    # MLP head — shared across both phases
    head      = MLP(input_dim=768, hidden_dims=hidden_dims, n_classes=n_classes, dropout_p=0.3).to(device)
    log.info(f"MLP architecture: 768 → {' → '.join(str(h) for h in hidden_dims)} → {n_classes}")
    criterion = nn.CrossEntropyLoss()

    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=["phase", "epoch", "test_top1", "test_top5", "test_loss", "lr"],
    )
    log_writer.writeheader()

    best_top1 = 0.0

    # ==================================================================
    # PHASE 1 — Head warmup on precomputed embeddings (MERT never runs)
    # ==================================================================
    if phase1_epochs > 0:
        log.info("=" * 60)
        log.info(f"PHASE 1: Head warmup on precomputed embeddings — {phase1_epochs} epochs")
        log.info("=" * 60)

        train_ds, test_ds = load_embedding_datasets(le)
        train_loader_p1   = DataLoader(train_ds, batch_size=PHASE1_BATCH, shuffle=True)
        test_loader_p1    = DataLoader(test_ds,  batch_size=PHASE1_BATCH, shuffle=False)

        optimizer_p1 = torch.optim.Adam(head.parameters(), lr=PHASE1_LR)
        scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_p1, mode="max", factor=PHASE1_FACTOR,
            patience=PHASE1_PATIENCE, min_lr=PHASE1_MIN_LR,
        )

        def train_step_p1(batch):
            head.train()
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_p1.zero_grad()
            criterion(head(X_batch), y_batch).backward()
            optimizer_p1.step()

        def save_p1(meta):
            torch.save({"head_state": head.state_dict(), "mert_state": None, **meta}, save_path)

        best_top1, _ = run_phase(
            phase_name="phase1",
            train_fn=train_step_p1,
            eval_fn=lambda loader, crit, dev: evaluate_embeddings(head, loader, crit, dev),
            train_loader=train_loader_p1,
            test_loader=test_loader_p1,
            optimizer=optimizer_p1,
            scheduler=scheduler_p1,
            criterion=criterion,
            n_epochs=phase1_epochs,
            device=device,
            log_writer=log_writer,
            save_fn=save_p1,
            best_top1=best_top1,
        )
        log.info(f"Phase 1 complete. Best top-1: {best_top1:.4f}")

    # ==================================================================
    # PHASE 2 — Fine-tuning top N MERT layers with frozen-layer caching
    # ==================================================================
    if phase2_epochs > 0:
        log.info("=" * 60)
        log.info(
            f"PHASE 2: Fine-tuning top {unfreeze_layers} MERT layers — "
            f"{phase2_epochs} epochs (with frozen-layer cache)"
        )
        log.info("=" * 60)

        log.info(f"Loading MERT: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        mert      = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

        model = MERTClassifier(mert, head, n_unfreeze=unfreeze_layers)

        # Load best Phase 1 head weights if checkpoint exists
        if save_path.exists():
            ckpt = torch.load(save_path, map_location=device)
            head.load_state_dict(ckpt["head_state"])
            log.info(f"Loaded Phase 1 head weights (best top-1: {ckpt['top1']:.4f})")

        freeze_mert(model)
        unfreeze_top_layers(model, unfreeze_layers)

        # Read manifest for train/test split
        with open(MANIFEST_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r["split"] == "train"]
        test_rows  = [r for r in rows if r["split"] == "test"]

        # ------------------------------------------------------------------
        # Build frozen-layer cache (one-time cost before training loop)
        # ------------------------------------------------------------------
        log.info("Building frozen-layer cache for train split...")
        train_cache = build_frozen_cache(
            mert, processor, train_rows, AUDIO_DIR, device, unfreeze_layers,
            cache_path=FINETUNE_DIR / f"frozen_cache_train_{unfreeze_layers}layers.pt",
            force_rebuild=force_rebuild_cache,
        )
        log.info("Building frozen-layer cache for test split...")
        test_cache  = build_frozen_cache(
            mert, processor, test_rows, AUDIO_DIR, device, unfreeze_layers,
            cache_path=FINETUNE_DIR / f"frozen_cache_test_{unfreeze_layers}layers.pt",
            force_rebuild=force_rebuild_cache,
        )

        train_dataset = CachedDataset(train_rows, train_cache, le)
        test_dataset  = CachedDataset(test_rows,  test_cache,  le)

        # Variant batch sampler — guarantees K variants per snippet per batch.
        # Falls back to standard shuffle if variants_per_snippet == 1.
        if variants_per_snippet > 1:
            train_sampler = VariantBatchSampler(
                train_dataset, batch_size=phase2_batch, K=variants_per_snippet, shuffle=True,
            )
            train_loader_p2 = DataLoader(
                train_dataset, batch_sampler=train_sampler,
                collate_fn=cached_collate_fn, num_workers=0,
            )
            log.info(
                f"VariantBatchSampler: K={variants_per_snippet}, "
                f"batch={phase2_batch}, aux_weight={aux_weight}"
            )
        else:
            train_loader_p2 = DataLoader(
                train_dataset, batch_size=phase2_batch, shuffle=True,
                collate_fn=cached_collate_fn, num_workers=0,
            )
            log.info("Standard shuffle sampler (variants_per_snippet=1, no aux loss)")

        test_loader_p2 = DataLoader(
            test_dataset, batch_size=phase2_batch, shuffle=False,
            collate_fn=cached_collate_fn, num_workers=0,
        )

        cos_sim = nn.CosineSimilarity(dim=1)

        mert_params  = [p for p in model.mert.parameters() if p.requires_grad]
        head_params  = list(model.head.parameters())
        optimizer_p2 = torch.optim.AdamW(
            [
                {"params": mert_params, "lr": PHASE2_LR_MERT},
                {"params": head_params, "lr": PHASE2_LR_HEAD},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        if phase2_scheduler == "cosine":
            scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_p2, T_max=phase2_epochs, eta_min=PHASE2_MIN_LR,
            )
            log.info(f"Phase 2 scheduler: CosineAnnealingLR (T_max={phase2_epochs})")
        else:
            scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_p2, mode="max", factor=PHASE2_FACTOR,
                patience=PHASE2_PATIENCE, min_lr=PHASE2_MIN_LR,
            )
            log.info("Phase 2 scheduler: ReduceLROnPlateau")

        def aux_cosine_loss(embeddings: torch.Tensor, snippet_keys: list) -> torch.Tensor:
            """
            Cosine similarity loss over variant pairs within the batch.

            For each unique snippet_key, finds all indices in the batch that
            share that key and computes pairwise (1 - cosine_similarity) for
            every pair. Returns the mean over all pairs found.
            If no pairs exist (e.g. variants_per_snippet=1), returns 0.
            """
            from collections import defaultdict
            key_to_indices = defaultdict(list)
            for i, key in enumerate(snippet_keys):
                key_to_indices[key].append(i)

            losses = []
            for indices in key_to_indices.values():
                if len(indices) < 2:
                    continue
                embs = embeddings[indices]          # (K, D)
                # All unique pairs (i, j), i < j
                for a in range(len(embs)):
                    for b in range(a + 1, len(embs)):
                        sim = cos_sim(embs[a].unsqueeze(0), embs[b].unsqueeze(0))
                        losses.append(1.0 - sim)

            if not losses:
                return torch.tensor(0.0, device=embeddings.device)
            return torch.stack(losses).mean()

        def train_step_p2(batch):
            model.train()
            hidden, mask, labels, snippet_keys = batch
            hidden = hidden.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)
            optimizer_p2.zero_grad()

            # Get pooled embeddings before the classifier head
            embeddings = model.get_embeddings(hidden, mask)
            logits     = model.head(embeddings)

            ce_loss  = criterion(logits, labels)
            aux_loss = aux_cosine_loss(embeddings, snippet_keys) if aux_weight > 0 else 0.0
            loss     = ce_loss + aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer_p2.step()

        def save_p2(meta):
            torch.save({
                "mert_state":      model.mert.state_dict(),
                "head_state":      model.head.state_dict(),
                "unfreeze_layers": unfreeze_layers,
                **meta,
            }, save_path)

        best_top1, _ = run_phase(
            phase_name="phase2",
            train_fn=train_step_p2,
            eval_fn=lambda loader, crit, dev: evaluate_cached(model, loader, crit, dev),
            train_loader=train_loader_p2,
            test_loader=test_loader_p2,
            optimizer=optimizer_p2,
            scheduler=scheduler_p2,
            criterion=criterion,
            n_epochs=phase2_epochs,
            device=device,
            log_writer=log_writer,
            save_fn=save_p2,
            best_top1=best_top1,
        )

    log_file.close()

    log.info("=" * 60)
    log.info("FINE-TUNING COMPLETE")
    log.info(f"  Best top-1:  {best_top1:.4f}  ({best_top1*100:.1f}%)")
    log.info(f"  Checkpoint:  {save_path}")
    log.info(f"  Log:         {log_path}")
    log.info(f"  Baseline:    34.4%  (frozen MERT + logistic regression)")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune MERT end-to-end on perturbed Bach chorales.")
    parser.add_argument("--phase1-epochs",         type=int,   default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs",         type=int,   default=PHASE2_EPOCHS)
    parser.add_argument("--unfreeze-layers",       type=int,   default=UNFREEZE_LAYERS)
    parser.add_argument("--hidden-dims",           type=int,   nargs="+", default=HIDDEN_DIMS,
                        help="MLP hidden layer sizes e.g. --hidden-dims 2048 512")
    parser.add_argument("--phase2-scheduler",      type=str,   default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler for phase 2 (default: cosine)")
    parser.add_argument("--variants-per-snippet",  type=int,   default=VARIANTS_PER_SNIPPET,
                        help="K variants per snippet per batch (1 = no pairing, default: 4)")
    parser.add_argument("--aux-weight",            type=float, default=AUX_WEIGHT,
                        help="Weight λ for cosine aux loss (0.0 = disabled, default: 0.3)")
    parser.add_argument("--phase2-batch",          type=int,   default=PHASE2_BATCH,
                        help="Phase 2 batch size (default: 128)")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Ignore existing cache checkpoints and rebuild from scratch")
    parser.add_argument("--device",              type=str,   default=EMBEDDING_DEVICE)
    args = parser.parse_args()

    run(
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        unfreeze_layers=args.unfreeze_layers,
        hidden_dims=args.hidden_dims,
        phase2_scheduler=args.phase2_scheduler,
        variants_per_snippet=args.variants_per_snippet,
        aux_weight=args.aux_weight,
        phase2_batch=args.phase2_batch,
        force_rebuild_cache=args.force_rebuild_cache,
        device=args.device,
    )
