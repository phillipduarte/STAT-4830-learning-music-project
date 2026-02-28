"""
embed.py — Stage 3 of the pipeline.

Responsibilities:
  1. Read the manifest CSV produced by data_prep.py.
  2. Load each WAV file rendered by render.py.
  3. Pass batches through the MERT transformer to get embeddings.
  4. Cache embeddings and labels to disk as numpy arrays.
  5. Skip already-embedded files (idempotent — safe to re-run).

Outputs (in data/embeddings/):
  embeddings_train.npy  — shape (N_train, embedding_dim)
  embeddings_test.npy   — shape (N_test,  embedding_dim)
  labels_train.npy      — shape (N_train,)  string labels
  labels_test.npy       — shape (N_test,)   string labels
  filenames_train.npy   — shape (N_train,)  original snippet filenames (for debugging)
  filenames_test.npy    — shape (N_test,)   original snippet filenames

Why separate train/test arrays:
  Keeps the split frozen from data_prep.py and makes classify.ipynb trivially simple —
  just np.load() and go. No risk of accidentally reshuffling the split.

MERT output:
  MERT returns hidden states of shape (batch, time_frames, hidden_dim).
  We mean-pool over the time dimension to get one vector per snippet.
  This is standard practice for audio classification with frame-level models.
"""

import csv
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SNIPPETS_DIR,
    AUDIO_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = SNIPPETS_DIR / "manifest.csv"


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
    """
    Load MERT (or any HuggingFace audio model) and its processor.

    The processor handles resampling and normalisation — it expects raw
    waveform arrays and returns tensors ready for the model.

    Swapping EMBEDDING_MODEL in config.py is all that's needed to try
    a different transformer here.
    """
    from transformers import AutoModel, AutoProcessor
    import torch

    log.info(f"Loading model: {model_id}  (device={device})")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model     = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model     = model.to(device)
    model.eval()  # disable dropout — we're doing inference only
    log.info("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------

def load_wav(wav_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """
    Load a WAV file as a 1D float32 numpy array at target_sr.

    Uses soundfile as the primary loader (fast, no ffmpeg needed).
    Falls back to librosa if soundfile fails (handles edge cases better).

    Returns None if loading fails entirely.
    """
    try:
        import soundfile as sf
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

        # Convert stereo to mono by averaging channels if necessary.
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Resample if FluidSynth produced a different rate than expected.
        # In practice this shouldn't happen since render.py passes -r explicitly,
        # but it's a good safety net.
        if sr != target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        return waveform

    except Exception as e_sf:
        log.debug(f"soundfile failed for {wav_path.name}, trying librosa: {e_sf}")
        try:
            import librosa
            waveform, _ = librosa.load(str(wav_path), sr=target_sr, mono=True)
            return waveform
        except Exception as e_lb:
            log.warning(f"Could not load {wav_path.name}: {e_lb}")
            return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(
    waveforms: list[np.ndarray],
    model,
    processor,
    device: str = EMBEDDING_DEVICE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Embed a batch of waveforms through MERT and return mean-pooled vectors.

    Args:
        waveforms:   List of 1D float32 numpy arrays (variable length is fine —
                     the processor pads them to the same length within the batch).
        model:       Loaded MERT model.
        processor:   Loaded MERT processor.
        device:      "cpu" or "cuda".
        sample_rate: Must match what the model was pretrained on (24kHz for MERT).

    Returns:
        numpy array of shape (len(waveforms), hidden_dim).

    Mean pooling explanation:
        MERT outputs hidden states at every time frame (like tokens in a text transformer).
        We average across all frames to get a single fixed-size vector per snippet.
        This is the simplest and most common aggregation strategy — alternatives
        like CLS token or attention pooling can be tried later.
    """
    import torch

    inputs = processor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,   # pad shorter waveforms in the batch to the longest
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use the last hidden state: shape (batch, time_frames, hidden_dim).
    # Mean pool over time_frames → shape (batch, hidden_dim).
    hidden = outputs.last_hidden_state          # (B, T, D)
    pooled = hidden.mean(dim=1)                 # (B, D)

    return pooled.cpu().numpy()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run(
    audio_dir: Path = AUDIO_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
    model_id: str = EMBEDDING_MODEL,
    device: str = EMBEDDING_DEVICE,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    force_reembed: bool = False,
) -> None:
    """
    Embed all WAV snippets and save results as numpy arrays.

    Args:
        audio_dir:      Directory of .wav files from render.py.
        embeddings_dir: Directory to write .npy embedding arrays.
        model_id:       HuggingFace model ID (swap in config.py to try others).
        device:         "cpu" or "cuda".
        batch_size:     Snippets per forward pass. Reduce if OOM.
        force_reembed:  If True, re-embed even if output files exist.
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete.
    output_files = [
        embeddings_dir / "embeddings_train.npy",
        embeddings_dir / "embeddings_test.npy",
        embeddings_dir / "labels_train.npy",
        embeddings_dir / "labels_test.npy",
    ]
    if all(f.exists() for f in output_files) and not force_reembed:
        log.info("Embeddings already exist. Use force_reembed=True to re-run.")
        return

    # Load manifest and partition by split.
    rows = _read_manifest(MANIFEST_PATH)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows  = [r for r in rows if r["split"] == "test"]
    log.info(f"Manifest loaded: {len(train_rows)} train, {len(test_rows)} test snippets.")

    # Load model once — expensive, don't do it inside the loop.
    model, processor = load_model(model_id, device)

    # Embed each split.
    for split_name, split_rows in [("train", train_rows), ("test", test_rows)]:
        log.info(f"Embedding {split_name} split ({len(split_rows)} snippets)...")

        all_embeddings = []
        all_labels     = []
        all_filenames  = []
        failed         = []

        # Process in batches.
        for batch_start in range(0, len(split_rows), batch_size):
            batch_rows = split_rows[batch_start : batch_start + batch_size]

            # Load waveforms for this batch.
            waveforms     = []
            valid_rows    = []

            for row in batch_rows:
                wav_name = Path(row["filename"]).with_suffix(".wav").name
                wav_path = audio_dir / wav_name
                waveform = load_wav(wav_path)
                if waveform is None:
                    failed.append(row["filename"])
                    continue
                waveforms.append(waveform)
                valid_rows.append(row)

            if not waveforms:
                continue

            # Forward pass through MERT.
            try:
                batch_embeddings = embed_batch(waveforms, model, processor, device)
                all_embeddings.append(batch_embeddings)
                all_labels.extend([r["label"] for r in valid_rows])
                all_filenames.extend([r["filename"] for r in valid_rows])
            except Exception as e:
                log.warning(f"Embedding failed for batch at index {batch_start}: {e}")
                failed.extend([r["filename"] for r in valid_rows])

            # Progress logging every 10 batches.
            batches_done = (batch_start // batch_size) + 1
            total_batches = (len(split_rows) + batch_size - 1) // batch_size
            if batches_done % 10 == 0 or batch_start + batch_size >= len(split_rows):
                log.info(f"  [{split_name}] batch {batches_done}/{total_batches} done")

        # Stack and save.
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)          # (N, D)
            labels_array     = np.array(all_labels, dtype=object) # (N,) strings
            filenames_array  = np.array(all_filenames, dtype=object)

            np.save(embeddings_dir / f"embeddings_{split_name}.npy", embeddings_array)
            np.save(embeddings_dir / f"labels_{split_name}.npy",     labels_array)
            np.save(embeddings_dir / f"filenames_{split_name}.npy",  filenames_array)

            log.info(f"  Saved {split_name}: shape={embeddings_array.shape}, failed={len(failed)}")
        else:
            log.error(f"No embeddings produced for {split_name} split!")

    log.info("=" * 50)
    log.info(f"Embeddings saved to: {embeddings_dir}")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run data_prep.py first."
        )
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed WAV snippets using MERT.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed even if output files already exist."
    )
    parser.add_argument(
        "--device", type=str, default=EMBEDDING_DEVICE,
        help="Device for inference: 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--batch-size", type=int, default=EMBEDDING_BATCH_SIZE,
        help="Snippets per forward pass. Reduce if out of memory."
    )
    args = parser.parse_args()

    run(force_reembed=args.force, device=args.device, batch_size=args.batch_size)
