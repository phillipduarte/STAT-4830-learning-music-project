"""
data_prep.py — Stage 1 of the pipeline.

Responsibilities:
  1. Load pieces from Music21 corpus or a directory of MIDI files.
  2. Chop each piece into fixed-length measure-based snippets.
  3. Save each snippet as an individual MIDI file with a self-describing filename.
  4. Produce a manifest CSV that maps every snippet to its piece label,
     composer, and train/test split assignment.

Output filename convention:
  {composer}__{piece_id}__s{snippet_index:04d}.mid
  e.g.  bach__bwv10.7__s0003.mid

The double-underscore separator makes it unambiguous to parse the filename
back into its components even if composer or piece_id contain single underscores.

The manifest CSV (data/snippets/manifest.csv) is the single source of truth
for labels and splits used by all downstream stages.
"""

import csv
import logging
import random
import re
from pathlib import Path
from typing import Optional

import music21
from music21 import corpus, converter, stream, midi

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_SOURCE,
    MUSIC21_CORPUS_COMPOSER,
    RAW_MIDI_DIR,
    SNIPPETS_DIR,
    SNIPPET_MEASURES,
    HOP_MEASURES,
    MIN_SNIPPETS_PER_PIECE,
    SPLIT_STRATEGY,
    TEST_SPLIT,
    RANDOM_SEED,
    LABEL_GRANULARITY,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pieces() -> list[dict]:
    """
    Load all pieces according to DATA_SOURCE config.

    Returns a list of dicts, each with keys:
      - "score":    music21.stream.Score
      - "composer": str
      - "piece_id": str  (unique within the dataset, filesystem-safe)
    """
    if DATA_SOURCE == "music21_bach":
        return _load_music21_bach()
    elif DATA_SOURCE == "music21_corpus":
        return _load_music21_composer(MUSIC21_CORPUS_COMPOSER)
    elif DATA_SOURCE == "midi_directory":
        return _load_midi_directory(RAW_MIDI_DIR)
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE!r}")


def _load_music21_bach() -> list[dict]:
    """Load all Bach chorales from Music21's built-in corpus."""
    log.info("Loading Bach chorales from Music21 corpus...")
    paths = corpus.getComposer("bach")
    pieces = []
    for path in paths:
        try:
            score = corpus.parse(path)
            piece_id = _path_to_piece_id(path)
            pieces.append({"score": score, "composer": "bach", "piece_id": piece_id})
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")
    log.info(f"Loaded {len(pieces)} Bach pieces.")
    return pieces


def _load_music21_composer(composer: str) -> list[dict]:
    """
    Load all works by a given composer from Music21's corpus.
    Switch DATA_SOURCE to 'music21_corpus' and set MUSIC21_CORPUS_COMPOSER
    in config.py to use this.
    """
    log.info(f"Loading {composer} from Music21 corpus...")
    paths = corpus.getComposer(composer)
    pieces = []
    for path in paths:
        try:
            score = corpus.parse(path)
            piece_id = _path_to_piece_id(path)
            pieces.append({"score": score, "composer": composer, "piece_id": piece_id})
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")
    log.info(f"Loaded {len(pieces)} pieces for composer '{composer}'.")
    return pieces


def _load_midi_directory(midi_dir: Path) -> list[dict]:
    """
    Load all .mid / .midi files from a directory.
    Composer is inferred as the parent folder name; piece_id is the stem.
    Expected structure (flat or one level deep):
        midi_dir/
            bach/piece1.mid
            beethoven/piece2.mid
        or just:
            midi_dir/piece1.mid  (composer = "unknown")
    """
    log.info(f"Loading MIDI files from {midi_dir}...")
    pieces = []
    for midi_file in sorted(midi_dir.rglob("*.mid")) + sorted(midi_dir.rglob("*.midi")):
        try:
            score = converter.parse(str(midi_file))
            composer = midi_file.parent.name if midi_file.parent != midi_dir else "unknown"
            piece_id = _sanitize_id(midi_file.stem)
            pieces.append({"score": score, "composer": composer, "piece_id": piece_id})
        except Exception as e:
            log.warning(f"Skipping {midi_file}: {e}")
    log.info(f"Loaded {len(pieces)} MIDI files.")
    return pieces


# ---------------------------------------------------------------------------
# Snippet Chopping
# ---------------------------------------------------------------------------

def chop_piece(score: music21.stream.Score, snippet_measures: int, hop_measures: int) -> list[music21.stream.Score]:
    """
    Chop a score into overlapping or non-overlapping measure-based windows.

    Args:
        score:            A music21 Score object.
        snippet_measures: Number of measures per snippet.
        hop_measures:     How many measures to advance per step.
                          hop_measures == snippet_measures → no overlap.
                          hop_measures < snippet_measures  → sliding overlap.

    Returns:
        List of Score objects, each containing snippet_measures measures.
    """
    # Flatten to a single part representation for measure extraction.
    # We keep ALL parts so the snippet retains full harmonic content.
    # We use measures() which is measure-number-aware and handles pickups correctly.
    try:
        # Get the measure range from the first part (all parts share measure numbers).
        first_part = score.parts[0]
        all_measures = first_part.getElementsByClass("Measure")
        measure_numbers = [m.number for m in all_measures]
    except (IndexError, AttributeError):
        log.warning("Could not extract measures — skipping piece.")
        return []

    if len(measure_numbers) < snippet_measures:
        return []

    snippets = []
    start_idx = 0

    while start_idx + snippet_measures <= len(measure_numbers):
        start_m = measure_numbers[start_idx]
        end_m   = measure_numbers[start_idx + snippet_measures - 1]

        # Extract the measure range across ALL parts to preserve harmony.
        try:
            snippet = score.measures(start_m, end_m)

            # measures() can return a Score or a Part depending on input —
            # normalise to Score so downstream code is consistent.
            if not isinstance(snippet, stream.Score):
                wrapper = stream.Score()
                wrapper.append(snippet)
                snippet = wrapper

            snippets.append(snippet)
        except Exception as e:
            log.warning(f"Failed to extract measures {start_m}-{end_m}: {e}")

        start_idx += hop_measures

    return snippets


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_snippet_as_midi(snippet: music21.stream.Score, out_path: Path) -> bool:
    """Write a single snippet Score to a MIDI file. Returns True on success."""
    try:
        mf = midi.translate.music21ObjectToMidiFile(snippet)
        mf.open(str(out_path), "wb")
        mf.write()
        mf.close()
        return True
    except Exception as e:
        log.warning(f"Could not save {out_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_PATH = SNIPPETS_DIR / "manifest.csv"
MANIFEST_COLUMNS = ["filename", "composer", "piece_id", "label", "snippet_index", "split"]


def _assign_splits_by_piece(pieces: list[dict], test_split: float, seed: int) -> dict[str, str]:
    """
    Assign train/test split at the PIECE level.
    Returns a dict mapping piece_id -> "train" | "test".

    Every snippet of a given piece lands entirely in one split.
    Use this for metric learning / open-set experiments where you want
    to test generalisation to genuinely unseen pieces.

    NOTE: incompatible with standard classifiers like logistic regression.
    Use SPLIT_STRATEGY = "by_snippet" for closed-set classification.
    """
    rng = random.Random(seed)
    piece_ids = [p["piece_id"] for p in pieces]
    rng.shuffle(piece_ids)
    n_test = max(1, int(len(piece_ids) * test_split))
    test_set = set(piece_ids[:n_test])
    return {pid: ("test" if pid in test_set else "train") for pid in piece_ids}


def _assign_splits_by_snippet(piece_id: str, n_snippets: int, test_split: float, seed: int) -> list:
    """
    Assign train/test split at the SNIPPET level for a single piece.
    Returns a list of split labels, one per snippet index.

    Every piece appears in BOTH train and test. The model sees some snippets
    of every piece during training and is evaluated on held-out snippets of
    the same pieces. Correct setup for closed-set classifiers (logistic
    regression, MLP) where the label space is fixed at training time.

    Uses a per-piece seed so results are reproducible regardless of
    the order in which pieces are processed.
    """
    rng = random.Random(f"{seed}_{piece_id}")
    indices = list(range(n_snippets))
    rng.shuffle(indices)
    n_test = max(1, int(n_snippets * test_split))
    test_indices = set(indices[:n_test])
    return ["test" if i in test_indices else "train" for i in range(n_snippets)]


def _make_label(piece: dict) -> str:
    """
    Construct the class label for a piece based on LABEL_GRANULARITY config.
    "piece"    → unique per piece  (e.g. "bach__bwv10.7")
    "composer" → one label per composer (e.g. "bach")
    """
    if LABEL_GRANULARITY == "piece":
        return f"{piece['composer']}__{piece['piece_id']}"
    elif LABEL_GRANULARITY == "composer":
        return piece["composer"]
    else:
        raise ValueError(f"Unknown LABEL_GRANULARITY: {LABEL_GRANULARITY!r}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run(
    snippet_measures: int = SNIPPET_MEASURES,
    hop_measures: int = HOP_MEASURES,
    min_snippets: int = MIN_SNIPPETS_PER_PIECE,
    split_strategy: str = SPLIT_STRATEGY,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
    out_dir: Path = SNIPPETS_DIR,
) -> Path:
    """
    Full data prep pipeline. Returns path to the manifest CSV.

    Accepts all config values as arguments so you can override them
    programmatically (e.g. from a notebook) without editing config.py.

    Args:
        split_strategy: "by_snippet" — each piece appears in both train and test
                                        (correct for closed-set classifiers).
                        "by_piece"   — whole pieces held out for test
                                        (correct for open-set / metric learning).
    """
    if split_strategy not in ("by_snippet", "by_piece"):
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}. Use 'by_snippet' or 'by_piece'.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all pieces.
    pieces = load_pieces()
    if not pieces:
        raise RuntimeError("No pieces loaded — check DATA_SOURCE config.")

    # 2. For by_piece strategy, pre-compute the piece-level split map.
    #    For by_snippet, splits are assigned per-piece inside the loop below.
    piece_split_map = None
    if split_strategy == "by_piece":
        piece_split_map = _assign_splits_by_piece(pieces, test_split, seed)
        log.info(f"Split strategy: by_piece — {sum(1 for v in piece_split_map.values() if v == 'test')} pieces held out for test.")
    else:
        log.info("Split strategy: by_snippet — every piece appears in both train and test.")

    # 3. Chop, save, and collect manifest rows.
    manifest_rows = []
    kept_pieces = 0
    skipped_pieces = 0

    for piece in pieces:
        composer  = piece["composer"]
        piece_id  = piece["piece_id"]
        label     = _make_label(piece)

        snippets = chop_piece(piece["score"], snippet_measures, hop_measures)

        if len(snippets) < min_snippets:
            log.debug(f"Skipping {piece_id}: only {len(snippets)} snippets (min={min_snippets}).")
            skipped_pieces += 1
            continue

        # Determine per-snippet split assignments.
        if split_strategy == "by_piece":
            # All snippets of this piece get the same split.
            snippet_splits = [piece_split_map[piece_id]] * len(snippets)
        else:
            # Each snippet is independently assigned; every piece gets both splits.
            snippet_splits = _assign_splits_by_snippet(piece_id, len(snippets), test_split, seed)

        for idx, (snippet, split) in enumerate(zip(snippets, snippet_splits)):
            filename = f"{composer}__{piece_id}__s{idx:04d}.mid"
            out_path = out_dir / filename

            if save_snippet_as_midi(snippet, out_path):
                manifest_rows.append({
                    "filename":      filename,
                    "composer":      composer,
                    "piece_id":      piece_id,
                    "label":         label,
                    "snippet_index": idx,
                    "split":         split,
                })

        kept_pieces += 1

    # 4. Write manifest CSV.
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 5. Summary.
    n_train = sum(1 for r in manifest_rows if r["split"] == "train")
    n_test  = sum(1 for r in manifest_rows if r["split"] == "test")
    n_labels = len(set(r["label"] for r in manifest_rows))

    log.info("=" * 50)
    log.info(f"Pieces kept:    {kept_pieces}  |  skipped: {skipped_pieces}")
    log.info(f"Total snippets: {len(manifest_rows)}  (train={n_train}, test={n_test})")
    log.info(f"Unique labels:  {n_labels}")
    log.info(f"Snippets saved to: {out_dir}")
    log.info(f"Manifest saved to: {MANIFEST_PATH}")
    log.info("=" * 50)

    return MANIFEST_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_piece_id(corpus_path) -> str:
    """
    Convert a Music21 corpus path object or string to a clean, filesystem-safe
    piece ID. Strips the file extension and sanitizes special characters.
    Example: 'bach/bwv10.7.mxl' → 'bwv10.7'
    """
    name = Path(str(corpus_path)).stem  # strip extension
    return _sanitize_id(name)


def _sanitize_id(s: str) -> str:
    """Replace characters that are unsafe in filenames with underscores."""
    # Allow alphanumeric, dots, hyphens. Replace everything else.
    return re.sub(r"[^\w.\-]", "_", s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    manifest = run()
    print(f"\nDone. Manifest at: {manifest}")
