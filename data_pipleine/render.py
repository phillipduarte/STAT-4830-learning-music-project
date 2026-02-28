"""
render.py — Stage 2 of the pipeline.

Responsibilities:
  1. Read the manifest CSV produced by data_prep.py.
  2. For each snippet MIDI, render it to a WAV file using FluidSynth.
  3. Skip snippets that have already been rendered (idempotent — safe to re-run).
  4. Write a render report summarising successes and failures.

Output:
  data/audio/{composer}__{piece_id}__s{idx:04d}.wav
  (mirrors the snippets directory structure, just different extension)

Dependencies:
  - FluidSynth must be installed system-level:
      macOS:  brew install fluidsynth
      Ubuntu: sudo apt install fluidsynth
  - A soundfont (.sf2) file must exist at SOUNDFONT_PATH in config.py.
    Recommended: GeneralUser GS (free, ~140MB)
    Download: https://www.polyphone-soundfonts.com/documents/27-instrument-sets/351-generaluser-gs-v1-471

Why subprocess over midi2audio:
  midi2audio is a thin wrapper that doesn't expose sample rate control.
  MERT requires exactly 24kHz mono audio — we enforce this explicitly
  via FluidSynth's -r flag so there's no silent sample rate mismatch.
"""

import csv
import logging
import subprocess
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SNIPPETS_DIR,
    AUDIO_DIR,
    SOUNDFONT_PATH,
    SAMPLE_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = SNIPPETS_DIR / "manifest.csv"


# ---------------------------------------------------------------------------
# Dependency Checks
# ---------------------------------------------------------------------------

def check_dependencies() -> None:
    """
    Verify FluidSynth binary and soundfont exist before starting a long job.
    Raises RuntimeError with a helpful message if anything is missing.
    """
    if shutil.which("fluidsynth") is None:
        raise RuntimeError(
            "FluidSynth binary not found.\n"
            "  macOS:  brew install fluidsynth\n"
            "  Ubuntu: sudo apt install fluidsynth\n"
        )

    if not SOUNDFONT_PATH.exists():
        raise RuntimeError(
            f"Soundfont not found at: {SOUNDFONT_PATH}\n"
            "Download GeneralUser GS from:\n"
            "  https://www.polyphone-soundfonts.com/documents/27-instrument-sets/351-generaluser-gs-v1-471\n"
            f"Then place it at: {SOUNDFONT_PATH}\n"
        )

    log.info("FluidSynth and soundfont found — dependencies OK.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_midi_to_wav(midi_path: Path, wav_path: Path, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Render a single MIDI file to WAV using FluidSynth.

    Args:
        midi_path:   Path to source .mid file.
        wav_path:    Path to write output .wav file.
        sample_rate: Output sample rate in Hz. Must match embedding model expectation.
                     MERT requires 24000.

    Returns:
        True on success, False on failure.

    FluidSynth flags used:
      -ni          non-interactive mode (no shell prompt)
      -F           output file path
      -r           sample rate
      -q           quiet (suppress FluidSynth's own logging to keep our logs clean)
    """
    cmd = [
        "fluidsynth",
        "-ni",                      # non-interactive
        "-q",                       # quiet FluidSynth's own output
        "-F", str(wav_path),        # output file
        "-r", str(sample_rate),     # sample rate — critical for MERT
        str(SOUNDFONT_PATH),        # soundfont
        str(midi_path),             # input MIDI
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60s timeout per snippet — should never be needed for short snippets
        )
        if result.returncode != 0:
            log.warning(f"FluidSynth failed for {midi_path.name}: {result.stderr.strip()}")
            return False
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            log.warning(f"FluidSynth produced empty file for {midi_path.name}")
            wav_path.unlink(missing_ok=True)
            return False
        return True

    except subprocess.TimeoutExpired:
        log.warning(f"FluidSynth timed out on {midi_path.name}")
        wav_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        log.warning(f"Unexpected error rendering {midi_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run(
    snippets_dir: Path = SNIPPETS_DIR,
    audio_dir: Path = AUDIO_DIR,
    sample_rate: int = SAMPLE_RATE,
    force_rerender: bool = False,
) -> None:
    """
    Render all MIDI snippets in the manifest to WAV files.

    Args:
        snippets_dir:   Directory containing .mid snippets and manifest.csv.
        audio_dir:      Directory to write .wav files.
        sample_rate:    Output sample rate (default from config — must match embedding model).
        force_rerender: If True, re-render even if WAV already exists.
                        If False (default), skip already-rendered files — safe to re-run.
    """
    check_dependencies()
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest to know which snippets to render.
    manifest_rows = _read_manifest(MANIFEST_PATH)
    if not manifest_rows:
        raise RuntimeError(f"Manifest is empty or missing: {MANIFEST_PATH}")

    total     = len(manifest_rows)
    skipped   = 0
    succeeded = 0
    failed    = 0

    log.info(f"Rendering {total} snippets to WAV at {sample_rate}Hz...")
    log.info(f"Output directory: {audio_dir}")

    for i, row in enumerate(manifest_rows, 1):
        midi_path = snippets_dir / row["filename"]
        wav_name  = Path(row["filename"]).with_suffix(".wav").name
        wav_path  = audio_dir / wav_name

        # Skip if already rendered (idempotent behaviour).
        if wav_path.exists() and wav_path.stat().st_size > 0 and not force_rerender:
            skipped += 1
            continue

        if i % 100 == 0 or i == total:
            log.info(f"  [{i}/{total}]  rendering {row['filename']}...")

        if not midi_path.exists():
            log.warning(f"MIDI file missing, skipping: {midi_path}")
            failed += 1
            continue

        if render_midi_to_wav(midi_path, wav_path, sample_rate):
            succeeded += 1
        else:
            failed += 1

    log.info("=" * 50)
    log.info(f"Rendered:  {succeeded}")
    log.info(f"Skipped (already exist): {skipped}")
    log.info(f"Failed:    {failed}")
    log.info(f"WAV files at: {audio_dir}")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: Path) -> list[dict]:
    """Read the manifest CSV into a list of row dicts."""
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

    parser = argparse.ArgumentParser(description="Render MIDI snippets to WAV via FluidSynth.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-render even if WAV files already exist."
    )
    args = parser.parse_args()

    run(force_rerender=args.force)
