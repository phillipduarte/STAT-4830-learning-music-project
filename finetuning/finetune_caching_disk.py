"""
finetune.py — End-to-end fine-tuning of MERT + sequence-aware head.

Three head architectures are supported via --head-type:
  mlp         Original: masked mean pool → BN → 512 → ReLU → Dropout → n_classes
  attn_pool   Learned attention pool → Linear → n_classes  (lightweight upgrade)
  lstm        Bidirectional LSTM → final hidden → Dropout → n_classes
  transformer Small Transformer encoder → masked mean pool → Linear → n_classes

Two-phase training strategy:
  Phase 1 — Warm up classifier head using precomputed embeddings (MERT never runs).
             Fast: identical to w7_mlp_search but using the best architecture.
             Requires data/embeddings/embeddings_train.npy to already exist.
             NOTE: Phase 1 is only compatible with --head-type mlp, because the
             precomputed embeddings are already mean-pooled (shape N×768). When
             using lstm / transformer / attn_pool, Phase 1 is automatically
             skipped and Phase 2 begins from a randomly initialised head.
  Phase 2 — Unfreeze top N transformer layers of MERT and train end-to-end,
             with frozen-layer output caching for a ~5x speedup.

Caching optimization (Phase 2):
  Only the top N MERT layers are unfrozen, so the bottom (12 - N) layers produce
  identical outputs every epoch. We precompute and cache their output once before
  Phase 2 begins, then each training step only runs the top N layers + head.
  This reduces MERT compute per batch from 12 layers down to N layers.

  Cache is stored as per-snippet tensors of shape (T_i, 768). By default the
  cache is saved to disk at data/finetune/frozen_cache_{split}_{N}layers.pt
  and reloaded on subsequent runs — so the cache-building cost is only paid
  once per (split, unfreeze_layers) combination. Changing unfreeze_layers
  produces a different filename so stale caches are never loaded by mistake.
  Pass --force-rebuild-cache to ignore existing files.

Best MLP architecture (from w7_mlp_search):
  768 → BN → 512 → ReLU → Dropout(0.3) → 430

Usage:
  python finetune.py
  python finetune.py --head-type lstm
  python finetune.py --head-type transformer
  python finetune.py --head-type attn_pool
  python finetune.py --phase1-epochs 30 --phase2-epochs 30 --unfreeze-layers 2
  python finetune.py --phase2-epochs 0   # Phase 1 only (mlp head only)
  python finetune.py --phase1-epochs 0   # Phase 2 only (loads existing checkpoint)

Outputs (saved to data/finetune/):
  finetune_best_{timestamp}.pt    — best checkpoint (MERT + head state dicts)
  finetune_log_{timestamp}.csv    — epoch-level training history
"""

import csv
import logging
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
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FINETUNE_DIR  = EMBEDDINGS_DIR.parent / "finetune"
MANIFEST_PATH = SNIPPETS_DIR / "manifest.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Phase 1 — head warmup on precomputed embeddings (fast; mlp head only)
PHASE1_EPOCHS   = 20
PHASE1_LR       = 1e-3
PHASE1_PATIENCE = 10
PHASE1_FACTOR   = 0.5
PHASE1_MIN_LR   = 1e-5
PHASE1_BATCH    = 256     # large is fine — no MERT forward pass

# Phase 2 — end-to-end fine-tuning with caching
PHASE2_EPOCHS   = 30
PHASE2_LR_HEAD  = 1e-3
PHASE2_LR_MERT  = 1e-5    # 100x lower — prevents catastrophic forgetting
PHASE2_PATIENCE = 10
PHASE2_FACTOR   = 0.5
PHASE2_MIN_LR   = 1e-6
PHASE2_BATCH    = 64      # can be larger now — only top N layers run per batch
WEIGHT_DECAY    = 1e-3

UNFREEZE_LAYERS = 2
LOG_EVERY       = 1
SEED            = 42

# Head hyperparameters
HEAD_TYPE           = "mlp"   # "mlp" | "attn_pool" | "lstm" | "transformer"
LSTM_HIDDEN_DIM     = 256     # per-direction; output is 2× this for bidirectional
LSTM_NUM_LAYERS     = 1
TRANSFORMER_NHEAD   = 8
TRANSFORMER_LAYERS  = 2
TRANSFORMER_FF_DIM  = 2048
HEAD_DROPOUT        = 0.3


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

    If cache_path is provided and the file exists (and force_rebuild is False),
    the cache is loaded from disk. The cache filename encodes n_unfreeze so
    changing that value never loads a stale cache — a new file is built automatically.

    Returns:
        cache: dict mapping filename -> (hidden_state, attention_mask)
            hidden_state:   float32 CPU tensor of shape (T_i, 768)
            attention_mask: bool CPU tensor of shape (T_i,), True = valid frame
    """
    # Load from disk if available
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        log.info(f"Loading frozen cache from disk: {cache_path.name}")
        cache = torch.load(cache_path, map_location="cpu")
        log.info(f"  Loaded {len(cache)} snippets from cache.")
        return cache

    # Build from scratch
    n_total_layers  = len(mert.encoder.layers)
    cache_layer_idx = n_total_layers - n_unfreeze
    log.info(
        f"Building frozen cache: layers 0-{cache_layer_idx - 1} "
        f"(top {n_unfreeze} layers run at train time)"
    )

    cache = {}
    mert.eval()

    with torch.no_grad():
        for i, row in enumerate(rows):
            wav_name  = Path(row["filename"]).with_suffix(".wav").name
            waveform  = _load_wav(audio_dir / wav_name)
            inputs    = processor(
                [waveform], sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=False,
            )
            inputs    = {k: v.to(device) for k, v in inputs.items()}
            outputs   = mert(**inputs, output_hidden_states=True)
            hidden    = outputs.hidden_states[cache_layer_idx].squeeze(0).cpu()
            attn_mask = torch.ones(hidden.shape[0], dtype=torch.bool)
            cache[row["filename"]] = (hidden, attn_mask)

            if (i + 1) % 200 == 0 or (i + 1) == len(rows):
                log.info(f"  Cache: {i+1}/{len(rows)} snippets processed")

    log.info(
        f"Cache built: {len(cache)} snippets, "
        f"~{sum(h.numel() * 4 for h, _ in cache.values()) / 1e9:.2f} GB"
    )

    # Save to disk for reuse on subsequent runs
    if cache_path is not None:
        torch.save(cache, cache_path)
        log.info(f"  Cache saved: {cache_path.name}")

    return cache


class CachedDataset(Dataset):
    """
    Dataset that returns pre-cached frozen-layer hidden states instead of raw audio.
    Each __getitem__ returns (hidden_state, attention_mask, label).
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
        row    = self.rows[idx]
        hidden, attn_mask = self.cache[row["filename"]]
        label  = int(self.le.transform([row["label"]])[0])
        return hidden, attn_mask, label


def cached_collate_fn(batch):
    """
    Collate variable-length cached hidden states into a padded batch.

    Pads shorter sequences to the max T in the batch with zero vectors,
    and sets the attention mask to False for padding frames.
    """
    hiddens, masks, labels = zip(*batch)
    max_T = max(h.shape[0] for h in hiddens)
    D     = hiddens[0].shape[1]

    padded_hidden = torch.zeros(len(hiddens), max_T, D)
    padded_mask   = torch.zeros(len(hiddens), max_T, dtype=torch.bool)

    for i, (h, m) in enumerate(zip(hiddens, masks)):
        T = h.shape[0]
        padded_hidden[i, :T] = h
        padded_mask[i,   :T] = m

    return padded_hidden, padded_mask, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Head architectures
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Original head: 768 → BN → 512 → ReLU → Dropout(0.3) → n_classes.

    Accepts sequences (B, T, 768) or pre-pooled vectors (B, 768).
    When given a sequence + mask it performs masked mean pooling internally,
    keeping Phase 1 (pre-pooled embeddings) and Phase 2 (sequences) compatible.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, n_classes),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            # Sequence input — masked mean pool to (B, D)
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        return self.net(x)


class AttentionPoolHead(nn.Module):
    """
    Lightweight upgrade over mean pooling: a learned scalar attention weight
    per frame (softmax-normalised, ignoring padding) collapses (B, T, D) → (B, D),
    then a single linear layer classifies.

    Adds only D + n_classes parameters over the raw MERT output — very low
    overfitting risk, but strictly more expressive than mean pooling because
    the model can up-weight the most discriminative frames.
    """

    def __init__(self, input_dim: int, n_classes: int, dropout_p: float = 0.1):
        super().__init__()
        self.attn   = nn.Linear(input_dim, 1)
        self.drop   = nn.Dropout(dropout_p)
        self.proj   = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)           # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)     # (B, T)
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return self.proj(self.drop(pooled))


class LSTMHead(nn.Module):
    """
    Bidirectional LSTM over the MERT frame sequence → classifier.

    Strategy: concatenate the final forward and backward hidden states of the
    last LSTM layer to obtain a fixed-size representation, then apply dropout
    and a linear projection.  Uses pack_padded_sequence / pad_packed_sequence
    so that padded frames do not affect the hidden-state trajectory.

    Args:
        input_dim:   Dimensionality of each input frame (768 for MERT-base).
        hidden_dim:  Hidden units per direction.  Output is 2 × hidden_dim.
        n_classes:   Number of output classes.
        num_layers:  Number of stacked LSTM layers.
        dropout_p:   Dropout applied between LSTM layers (ignored for num_layers=1)
                     and before the final linear projection.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        n_classes:  int,
        num_layers: int   = 1,
        dropout_p:  float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout_p)
        self.proj = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, input_dim) — padded sequence of MERT frame embeddings.
            mask: (B, T) bool        — True for real frames, False for padding.

        Returns:
            logits: (B, n_classes)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()   # (B,) — true lengths
            packed  = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(x)

        # hn: (num_layers * 2, B, hidden_dim) for a bidirectional LSTM.
        # The last layer's forward and backward hidden states are at indices
        # -2 and -1 respectively.
        forward_h  = hn[-2]                                # (B, hidden_dim)
        backward_h = hn[-1]                                # (B, hidden_dim)
        pooled = torch.cat([forward_h, backward_h], dim=-1)  # (B, 2 * hidden_dim)
        return self.proj(self.drop(pooled))


class TransformerHead(nn.Module):
    """
    Small Transformer encoder on top of MERT frame embeddings → classifier.

    Note: MERT is already a large transformer, so this head is best thought of
    as a lightweight re-weighting / interaction layer rather than a deep model.
    Keep num_layers small (1–2) to avoid overfitting and redundancy.

    The output is mean-pooled over valid frames (padding excluded) before
    the final linear projection.  No CLS token is added — masked mean pooling
    over a transformer's output is empirically competitive and simpler.

    Args:
        input_dim:  Feature dimension of each frame (768 for MERT-base).
        n_classes:  Number of output classes.
        nhead:      Number of self-attention heads (must divide input_dim evenly).
        num_layers: Number of TransformerEncoderLayer blocks (recommend 1–2).
        ff_dim:     Feed-forward hidden dimension inside each layer.
        dropout_p:  Dropout in attention and feed-forward layers.
    """

    def __init__(
        self,
        input_dim:  int,
        n_classes:  int,
        nhead:      int   = 8,
        num_layers: int   = 2,
        ff_dim:     int   = 2048,
        dropout_p:  float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout_p,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout_p)
        self.proj = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, input_dim)
            mask: (B, T) bool — True = real frame, False = padding.
                  Converted to src_key_padding_mask (True = ignore) for PyTorch.

        Returns:
            logits: (B, n_classes)
        """
        # nn.TransformerEncoder expects src_key_padding_mask where True = *ignore*
        padding_mask = ~mask if mask is not None else None   # (B, T)
        out = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T, D)

        # Masked mean pool over valid frames
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()              # (B, T, 1)
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        return self.proj(self.drop(pooled))


def build_head(
    head_type:   str,
    input_dim:   int,
    n_classes:   int,
    dropout_p:   float = HEAD_DROPOUT,
    lstm_hidden: int   = LSTM_HIDDEN_DIM,
    lstm_layers: int   = LSTM_NUM_LAYERS,
    tf_nhead:    int   = TRANSFORMER_NHEAD,
    tf_layers:   int   = TRANSFORMER_LAYERS,
    tf_ff_dim:   int   = TRANSFORMER_FF_DIM,
) -> nn.Module:
    """Factory that instantiates the requested head architecture."""
    head_type = head_type.lower()
    if head_type == "mlp":
        return MLP(input_dim, 512, n_classes, dropout_p)
    elif head_type == "attn_pool":
        return AttentionPoolHead(input_dim, n_classes, dropout_p)
    elif head_type == "lstm":
        return LSTMHead(input_dim, lstm_hidden, n_classes, lstm_layers, dropout_p)
    elif head_type == "transformer":
        return TransformerHead(input_dim, n_classes, tf_nhead, tf_layers, tf_ff_dim, dropout_p)
    else:
        raise ValueError(f"Unknown head type '{head_type}'. Choose: mlp, attn_pool, lstm, transformer")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MERTClassifier(nn.Module):
    """
    MERT backbone + pluggable sequence head.

    Supports two forward modes:
      forward(inputs)                     — full pass through all MERT layers (cache build)
      forward_from_cache(hidden, mask)    — top N unfrozen layers only (Phase 2 training)

    The head receives (hidden, mask) in both cases, where hidden is (B, T, 768)
    and mask is (B, T) bool.  Each head implementation handles pooling internally,
    so MERTClassifier no longer performs any pooling itself.
    """

    def __init__(self, mert_model, head: nn.Module, n_unfreeze: int):
        super().__init__()
        self.mert       = mert_model
        self.head       = head
        self.n_unfreeze = n_unfreeze

    def forward(self, inputs: dict) -> torch.Tensor:
        """Full forward pass — used for cache building only."""
        outputs = self.mert(**inputs, output_hidden_states=False)
        hidden  = outputs.last_hidden_state                     # (B, T, 768)
        mask    = inputs.get("attention_mask")                  # (B, T) or None
        if mask is not None:
            mask = mask.bool()
        return self.head(hidden, mask)

    def forward_from_cache(
        self,
        cached_hidden:  torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run only the top N unfrozen MERT layers on a cached hidden state, then head.

        Args:
            cached_hidden:  (B, T, 768) — output of last frozen layer, on device
            attention_mask: (B, T) bool — True for valid frames, False for padding

        Returns:
            logits: (B, n_classes)

        The attention mask is converted to an additive mask (0 for valid, -1e4 for
        padding) in the format expected by HuBERT-style transformer encoder layers.
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

        # Delegate all pooling / sequence modelling to the head
        return self.head(hidden, attention_mask)


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

def evaluate_embeddings(head: nn.Module, loader: DataLoader, criterion, device: str) -> dict:
    """
    Phase 1 eval — head only on precomputed (pre-pooled) embeddings.
    Only valid for the MLP head; other heads receive sequences, not pooled vectors.
    """
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
        for hidden, mask, labels in loader:
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
        scheduler.step(metrics["top1"])

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
    phase1_epochs:       int  = PHASE1_EPOCHS,
    phase2_epochs:       int  = PHASE2_EPOCHS,
    unfreeze_layers:     int  = UNFREEZE_LAYERS,
    device:              str  = EMBEDDING_DEVICE,
    model_id:            str  = EMBEDDING_MODEL,
    force_rebuild_cache: bool = False,
    head_type:           str  = HEAD_TYPE,
) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FINETUNE_DIR / f"finetune_best_{run_id}.pt"
    log_path  = FINETUNE_DIR / f"finetune_log_{run_id}.csv"
    log.info(f"Run ID: {run_id}  |  Head type: {head_type}")

    # Phase 1 is only compatible with the MLP head (precomputed embeddings are pooled).
    # For sequence heads (lstm, transformer, attn_pool) silently skip Phase 1.
    if head_type != "mlp" and phase1_epochs > 0:
        log.warning(
            f"Phase 1 requires pre-pooled embeddings and is incompatible with "
            f"head_type='{head_type}'. Skipping Phase 1 automatically."
        )
        phase1_epochs = 0

    # Label encoder — matches existing pipeline label→index mapping
    y_train_str = np.load(EMBEDDINGS_DIR / "labels_train.npy", allow_pickle=True)
    y_test_str  = np.load(EMBEDDINGS_DIR / "labels_test.npy",  allow_pickle=True)
    le = LabelEncoder()
    le.fit(y_train_str)
    n_classes = len(le.classes_)
    log.info(f"Classes: {n_classes}  |  Device: {device}")

    # Shared head — built once and used across both phases
    head      = build_head(head_type, input_dim=768, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=["phase", "epoch", "test_top1", "test_top5", "test_loss", "lr"],
    )
    log_writer.writeheader()

    best_top1 = 0.0

    # ==================================================================
    # PHASE 1 — Head warmup on precomputed embeddings (MLP only)
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
        # Build frozen-layer cache (loaded from disk if already exists)
        # Cache filename encodes unfreeze_layers — changing it auto-invalidates.
        # ------------------------------------------------------------------
        train_cache_path = FINETUNE_DIR / f"frozen_cache_train_{unfreeze_layers}layers.pt"
        test_cache_path  = FINETUNE_DIR / f"frozen_cache_test_{unfreeze_layers}layers.pt"

        log.info("Loading/building frozen-layer cache for train split...")
        train_cache = build_frozen_cache(
            mert, processor, train_rows, AUDIO_DIR, device, unfreeze_layers,
            cache_path=train_cache_path, force_rebuild=force_rebuild_cache,
        )
        log.info("Loading/building frozen-layer cache for test split...")
        test_cache = build_frozen_cache(
            mert, processor, test_rows, AUDIO_DIR, device, unfreeze_layers,
            cache_path=test_cache_path, force_rebuild=force_rebuild_cache,
        )

        train_loader_p2 = DataLoader(
            CachedDataset(train_rows, train_cache, le),
            batch_size=PHASE2_BATCH, shuffle=True,
            collate_fn=cached_collate_fn, num_workers=0,
        )
        test_loader_p2 = DataLoader(
            CachedDataset(test_rows, test_cache, le),
            batch_size=PHASE2_BATCH, shuffle=False,
            collate_fn=cached_collate_fn, num_workers=0,
        )

        mert_params  = [p for p in model.mert.parameters() if p.requires_grad]
        head_params  = list(model.head.parameters())
        optimizer_p2 = torch.optim.AdamW(
            [
                {"params": mert_params, "lr": PHASE2_LR_MERT},
                {"params": head_params, "lr": PHASE2_LR_HEAD},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_p2, mode="max", factor=PHASE2_FACTOR,
            patience=PHASE2_PATIENCE, min_lr=PHASE2_MIN_LR,
        )

        def train_step_p2(batch):
            model.train()
            hidden, mask, labels = batch
            hidden = hidden.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)
            optimizer_p2.zero_grad()
            loss = criterion(model.forward_from_cache(hidden, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer_p2.step()

        def save_p2(meta):
            torch.save({
                "mert_state":      model.mert.state_dict(),
                "head_state":      model.head.state_dict(),
                "head_type":       head_type,
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
    log.info(f"  Head type:   {head_type}")
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

    parser = argparse.ArgumentParser(description="Fine-tune MERT end-to-end on Bach chorales.")
    parser.add_argument("--phase1-epochs",       type=int,  default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs",       type=int,  default=PHASE2_EPOCHS)
    parser.add_argument("--unfreeze-layers",     type=int,  default=UNFREEZE_LAYERS)
    parser.add_argument("--device",              type=str,  default=EMBEDDING_DEVICE)
    parser.add_argument(
        "--head-type",
        type=str,
        default=HEAD_TYPE,
        choices=["mlp", "attn_pool", "lstm", "transformer"],
        help=(
            "Classification head architecture. "
            "'mlp': mean pool → MLP (original). "
            "'attn_pool': learned attention pool → linear. "
            "'lstm': bidirectional LSTM → final hidden → linear. "
            "'transformer': small Transformer encoder → mean pool → linear. "
            "Note: lstm/transformer/attn_pool automatically skip Phase 1."
        ),
    )
    parser.add_argument("--lstm-hidden",         type=int,  default=LSTM_HIDDEN_DIM,
                        help="Hidden units per direction for the LSTM head.")
    parser.add_argument("--lstm-layers",         type=int,  default=LSTM_NUM_LAYERS,
                        help="Number of LSTM layers.")
    parser.add_argument("--tf-nhead",            type=int,  default=TRANSFORMER_NHEAD,
                        help="Attention heads for the Transformer head.")
    parser.add_argument("--tf-layers",           type=int,  default=TRANSFORMER_LAYERS,
                        help="Number of Transformer encoder layers.")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Ignore existing disk cache and rebuild from audio.")
    args = parser.parse_args()

    run(
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        unfreeze_layers=args.unfreeze_layers,
        device=args.device,
        force_rebuild_cache=args.force_rebuild_cache,
        head_type=args.head_type,
    )
