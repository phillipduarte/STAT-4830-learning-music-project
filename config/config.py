"""
config.py — Central configuration for the music classification pipeline.

All paths, hyperparameters, and model choices live here.
Changing one value here propagates through the entire pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

RAW_MIDI_DIR    = DATA_DIR / "raw"        # original MIDI files (if using external dataset)
SNIPPETS_DIR    = DATA_DIR / "snippets"   # chopped MIDI snippets
AUDIO_DIR       = DATA_DIR / "audio"      # WAV renderings of snippets (Tier 2)
EMBEDDINGS_DIR  = DATA_DIR / "embeddings" # cached numpy embedding arrays

# ---------------------------------------------------------------------------
# Data Source
# ---------------------------------------------------------------------------

# "music21_bach"      — use Music21's built-in Bach chorales (recommended for baseline)
# "music21_corpus"    — use any Music21 corpus (set MUSIC21_CORPUS_PATHS below)
# "midi_directory"    — load raw MIDI files from RAW_MIDI_DIR
DATA_SOURCE = "music21_bach"

# Only used when DATA_SOURCE == "music21_corpus".
# These are Music21 corpus path prefixes — e.g. "beethoven/opus18no1" or "palestrina"
# Run `music21.corpus.getComposer('beethoven')` to see available paths.
MUSIC21_CORPUS_COMPOSER = "bach"  # e.g. "beethoven", "palestrina", "handel"

# ---------------------------------------------------------------------------
# Snippet Parameters
# ---------------------------------------------------------------------------

SNIPPET_MEASURES = 4   # length of each snippet in measures

# Set HOP_MEASURES == SNIPPET_MEASURES for zero overlap (matches real-world "page" use case).
# Set HOP_MEASURES < SNIPPET_MEASURES for sliding window overlap (more training data).
# Recommended: 2 for training data augmentation, but we split train/test by PIECE
# so overlapping snippets from the same piece never straddle the split boundary.
HOP_MEASURES = 2

# Discard any piece that yields fewer than this many snippets after chopping.
# Prevents classes with almost no training data.
MIN_SNIPPETS_PER_PIECE = 3

# How to assign train/test splits:
#
# "by_snippet" — (default) for each piece, ~TEST_SPLIT fraction of its snippets
#                go to test and the rest to train. Every piece appears in BOTH
#                splits. Use this for the logistic regression baseline and any
#                closed-set classifier — the model can be fairly evaluated on
#                held-out snippets of pieces it has seen during training.
#
# "by_piece"   — entire pieces are held out for test. No snippets of a test
#                piece are ever seen during training. Use this for metric
#                learning / triplet loss experiments where you want to test
#                generalisation to genuinely unseen pieces.
#
# Switching between these requires re-running data_prep.py to regenerate the
# manifest, then re-running embed.py to regenerate the embedding arrays.
SPLIT_STRATEGY = "by_snippet"

# Fraction of data held out for testing.
# For "by_snippet": fraction of each piece's snippets go to test.
# For "by_piece":   fraction of pieces go to test entirely.
TEST_SPLIT = 0.2

# Random seed for reproducibility across train/test splits.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Label Granularity
# ---------------------------------------------------------------------------

# "piece"    — each individual piece is its own class (~400 classes for Bach)
# "composer" — each composer is a class (easy baseline sanity check)
LABEL_GRANULARITY = "piece"

# ---------------------------------------------------------------------------
# Audio Rendering (Tier 2: MIDI → WAV via FluidSynth)
# ---------------------------------------------------------------------------

# Download GeneralUser GS soundfont from:
# https://www.polyphone-soundfonts.com/documents/27-instrument-sets/351-generaluser-gs-v1-471
SOUNDFONT_PATH = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")

# MERT expects 24kHz mono audio — do not change unless switching embedding models.
SAMPLE_RATE = 24_000

# ---------------------------------------------------------------------------
# Embedding Model (Tier 2)
# ---------------------------------------------------------------------------

# HuggingFace model ID. Swap this string to try different models.
# "m-a-p/MERT-v1-95M"  — 95M param music audio transformer (recommended)
# "m-a-p/MERT-v1-330M" — larger, slower, potentially better
EMBEDDING_MODEL = "m-a-p/MERT-v1-95M"

# Device for embedding inference. "cuda" if GPU available, else "cpu".
EMBEDDING_DEVICE = "cpu"

# How many snippet WAVs to embed in one forward pass.
# Reduce if you run out of memory.
EMBEDDING_BATCH_SIZE = 8
