"""
perturb.py — Separate stage to make alterations to WAV files (rather than music21 Scores/MIDI)

update 4/3/2026: There are issues in data leakage due to the use of
_assign_splits_by_snippet in the last part (5) of the perturb.py where this meant
that the snippet that was perturbed (ex. 95% speed or 105% speed) were allowed 
to be in both the train/test sets - leading to misleading results.
-> To fix this, now need to perform the split before the perturbations.
-> When creating the 8s snippets, also changing the hop to 6s instead of 4s
(so adj snippets have 2s overlap) thus we modify. Also modified to only keep
snippets with length >= 6s (before perturb)
3. Load WAV files just saved and create snippets (see hop_seconds) of the
waveforms and perturb them before saving snippet WAV files. Assign snippets
for each 'piece' (i.e. a perturbation) to create the manifest.csv

Responsibilities:
  1. Load MIDI files of interest
    a. From music21 directly -> convert to midi file
    b. Direct midi -> just load
  2. Convert MIDI files to WAV files
  3. Create perturbations in 2 ways: 
    a. Tempo change (slower/faster, via librosa)
    b. Pitch shift (different key, via librosa)
  4. Create snippets non-overlap and overlap (see: HOP_MEASURES), save as WAV
  5. Also create the manifest file with train/test splits.
  (afterwards) Embed these WAV snippets and save as .npy (same process as embed.py)
"""
import logging
import csv
from pathlib import Path
import numpy as np
import random
import soundfile as sf
import librosa

# Import data_prep.py so we can use "_load_music21_composer"
import data_prep
# Import render.py to help with MIDI -> WAV
import render
# Import embed.py to use load_wav
import embed
# NOTE: SOUNDFONT_PATH is used in render.py functions
from config import (
    PERTURB_IS_M21,
    PERTURB_COMPOSER,
    PERTURB_DIR,
    MIDI_DIR,
    SAMPLE_RATE,
    SOUNDFONT_PATH,
    MIN_SNIPPETS_PER_PIECE,
    TEST_SPLIT,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Loading MIDI
# ---------------------------------------------------------------------------
def load_music21(out_dir: Path) -> list[dict]:
    """
    Load music21 pieces according to PERTURB_COMPOSER config (scroll to bottom)
    Then, convert to MIDI and save files

    Returns a list of dicts, each with keys:
      - "midi_path": Path
      - "composer": str
      - "piece_id": str  (unique within the dataset, filesystem-safe)
    """
    # This is list[dict] of same as above but instaed of midi_path we have "score": music21.stream.Score
    music21_streams = data_prep._load_music21_composer(PERTURB_COMPOSER)
    
    # See 'save_snippet_as_midi' from data_prep.py, iterate through all pieces
    m21_midis = list()
    for piece in music21_streams:
        filename = f"{piece['composer']}__{piece['piece_id']}.mid"
        out_path = out_dir / filename
        success = data_prep.save_snippet_as_midi(piece['score'], out_path)
        if success:
            m21_midis.append({'midi_path': out_path,
                                'composer': piece['composer'],
                                'piece_id': piece['piece_id']})

    # Now return the list[dict] with these path names
    return m21_midis

def load_midi(midi_dir: Path) -> list[dict]:
    """
    Load direct MIDI files to a list[dict] with same type as load_music21 above
    TODO: rename all the symphonies to have composer + "__" + piece_id
    """
    midis = list()
    folder = Path(midi_dir)
    for piece_path in folder.iterdir():
        # stem removes the ".mid"
        midis.append({'midi_path': piece_path,
                      'composer': piece_path.stem,
                      'piece_id': ""})
    return midis

# ---------------------------------------------------------------------------
# 2. Convert MIDI to WAV - uses the render.py stuff entirely, see run()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Create perturbations (result in original, tempo change, pitch shift versions)
# ---------------------------------------------------------------------------
def load_wav_files(wav_dir: Path) -> list[dict]:
    """
    Load WAV files to a list[dict] with the same type as load_midi except
    we have the wav_path rather than the midi path
    NOTE: not to be confused with load_wav in embed.py
    """
    wavs = list()
    folder = Path(wav_dir)
    for piece_path in folder.iterdir():
        # stem removes the ".wav"
        wavs.append({'wav_path': piece_path,
                      'composer': piece_path.stem,
                      'piece_id': ""})
    return wavs

def perturb(out_dir: Path, wav_path: Path, waveform: np.ndarray) -> None:
    """
    filename: like "bach__bwv1.6", waveform: 1D float32 numpy array
    Perturbs the wav_path found at the path and saves to a new folder pwav ("perturbed wav")
    a) 11 tempo changes: 0 (original), +/- 5% (1.05, 0.95), +/- 10% (1.1, 0.9),
        +/- 15% (1.15, 0.85), +/- 20% (1.2, 0.8), +/- 25% (1.25, 0.75)
    b) 11 pitch shifts: 0 (original), +/- 12 (octaves), +/- 1 (1 semitone up/down),
        +/- 2 (2 semitones up/down), +/- 3 (3 semitones up/down) +5 (4th), +7 (5th)
    Combinations of these gives us 120 total perturbations + 1 original = 121
    """
    # tempos = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    # pitches = [0, -1, 1, -2, 2, -3, 3, 4, 5, -12, 12]
    # Smaller version
    tempos = [0.9, 0.95, 1.0, 1.05, 1.1]
    pitches = [0, -1, 1, -2, 2]
    for t in tempos:
        for p in pitches:
            wf = waveform.copy()
            # This will say something like "bach__001__t1p0.wav"
            combo = "__t" + str(t) + "p" + str(p)
            out_path = out_dir / f"{wav_path.stem}{combo}.wav"
            if t != 1.0:
                wf = librosa.effects.time_stretch(y=wf, rate=t)
            if p != 0:
                wf = librosa.effects.pitch_shift(y=wf, sr=SAMPLE_RATE, n_steps=p)
            # The default 'subtype' is 32-bit float "FLOAT"
            sf.write(out_path, wf, SAMPLE_RATE, format='WAV')

def perturb_snippet(out_dir: Path, wav_path: Path, waveform: np.ndarray, snippet_idx: int) -> None:
    """
    Save as above but for a snippet instead of a full piece
    That is, we include the snippet index
    """
    # tempos = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    # pitches = [0, -1, 1, -2, 2, -3, 3, 4, 5, -12, 12]
    # Smaller version
    # tempos = [0.9, 0.95, 1.0, 1.05, 1.1]
    tempos = [0.8, 0.9, 1.0, 1.1, 1.2] # Changing this!
    # pitches = [0, -1, 1, -2, 2]
    # I am changing the pitches to be more different (4 = major 3rd, 5 = 4th, 7 = 5th)
    pitches = [-3, 3, -4, 4]
    for t in tempos:
        for p in pitches:
            wf = waveform.copy()
            # This will say something like "bach__001__t1p0__s0001.wav"
            perturbation = "t" + str(t) + "p" + str(p)
            out_path = out_dir / f"{wav_path.stem}__{perturbation}__s{snippet_idx:04d}.wav"
            if t != 1.0:
                wf = librosa.effects.time_stretch(y=wf, rate=t)
            if p != 0:
                wf = librosa.effects.pitch_shift(y=wf, sr=SAMPLE_RATE, n_steps=p)
            # The default 'subtype' is 32-bit float "FLOAT"
            sf.write(out_path, wf, SAMPLE_RATE, format='WAV')

# ---------------------------------------------------------------------------
# 4. Create snippets from WAVs based on duration (overlapping/non-overlapping)
# ---------------------------------------------------------------------------
def chop_piece_wav(wav_path: Path, snippet_seconds: int, hop_seconds: int) -> list[np.ndarray]:
    """
    See 'chop_piece' in data_prep.py, difference is that we chop the WAVs directly
    based on duration. hop_seconds as the amt of overlap between snippets.
    Returning as list[np.ndarray] (we will save to WAV separately in the run())

    Most Bach snippets around 8s, though I saw 5s, 12s
    """
    # y = waveform np.ndarray, sr = sample_rate
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    snippet_samples = snippet_seconds * sr
    snippets = []
    start_idx = 0
    # This while condition enforces that snippets are 8s
    while start_idx + snippet_samples <= len(y):
        snippet = y[start_idx:start_idx + snippet_samples]
        snippets.append(snippet)
        start_idx += hop_seconds * sr
    return snippets

def _assign_splits_by_unperturbed_snippet(piece_id:str, n_snippets: int, test_split: float, seed: int) -> list:
    """
    Modeled after _assign_splits_by_snippet in data_prep.py
    Return: list["test", "train", ...] etc. where list[0] = "test" means the snippet
    at index 0 is in the test set
    """
    rng = random.Random(f"{seed}_{piece_id}")
    indices = list(range(n_snippets))
    rng.shuffle(indices)
    n_test = max(1, int(n_snippets * test_split))
    test_indices = set(indices[:n_test])
    return ["test" if i in test_indices else "train" for i in range(n_snippets)]

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run(
    perturb_dir: Path = PERTURB_DIR,
    midi_dir: Path = MIDI_DIR,
    sample_rate: int = SAMPLE_RATE,
    force_rerender: bool = False,
    min_snippets: int = MIN_SNIPPETS_PER_PIECE,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED
) -> None:
    """
    Full pipeline for perturbations
    """
    # 1. Load all pieces as MIDI files in perturb/midi
    out_dir = perturb_dir / "midi"
    out_dir.mkdir(parents=True, exist_ok=True)
    if PERTURB_IS_M21:
        midis = load_music21(out_dir)
    else:
        midis = load_midi(midi_dir)
    if not midis:
        raise RuntimeError("No pieces loaded.") 

    # 2. (see render.py) Convert MIDI -> WAV for files we just processed
    render.check_dependencies()
    wav_dir = perturb_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    i = succeeded = failed_render = skipped = 0
    # total = len(midis)
    for m in midis:
        midi_path = m["midi_path"]
        wav_path = wav_dir / midi_path.with_suffix(".wav").name

        # Skip if already rendered (idempotent behaviour).
        if wav_path.exists() and wav_path.stat().st_size > 0 and not force_rerender:
            skipped += 1
            continue

        # if i % 100 == 0 or i == total:
        #     log.info(f"  [{i}/{total}]  rendering {midi_path.name}...")

        if not midi_path.exists():
            log.warning(f"MIDI file missing, skipping: {midi_path}")
            failed_render += 1
            continue
    
        if render.render_midi_to_wav(midi_path, wav_path, sample_rate):
            succeeded += 1
        else:
            failed_render += 1
    log.info("=" * 50)
    log.info(f"Rendered:  {succeeded}")
    log.info(f"Skipped (already exist): {skipped}")
    log.info(f"Failed:    {failed_render}")
    log.info(f"WAV files at: {wav_dir}")
    log.info("=" * 50)
    
    # 3. Go through WAV files in wav_dir and create snippets, then immediately
    # perturb them, then treating each of those perturbations as a 'piece',
    # assign to the train/test and create the manifest.csv
    folder = Path(wav_dir)
    psnippets_new1_dir = perturb_dir / "psnippets_new1"
    psnippets_new1_dir.mkdir(parents=True, exist_ok=True)
    pmanifest_rows = []
    kept_pieces = 0
    skipped_pieces = []

    # These are the perturbations used in filenames (in each iter I ran this)
    # (0) tempos = [0.9, 0.95, 1.0, 1.05, 1.1], pitches = [0, -1, 1, -2, 2]
    # (1) tempos = [0.8, 0.9, 1.0, 1.1, 1.2], pitches = [0, -1, 1, -2, 2]
    # (2)
    tempos = [0.8, 0.9, 1.0, 1.1, 1.2]
    pitches = [-3, 3, -4, 4]
    perturbs = ["t" + str(t) + "p" + str(p) for t in tempos for p in pitches]

    for piece_path in folder.iterdir():
        # This returns a list[np.ndarray]
        snip_waveforms = chop_piece_wav(piece_path, snippet_seconds=8, hop_seconds=6)
        if len(snip_waveforms) < min_snippets:
            log.debug(f"Skipping {piece_path}: only {len(snip_waveforms)} snippets (min={min_snippets}).")
            skipped_pieces.append(piece_path)
            continue
        # At this point we should assign the splits so that for each unperturbed
        # snippet, it is either in train/test, and all perturbations of a snippet too
        # Paths have form 'bach__bwv404.wav' (stem = omits .wav)
        composer = piece_path.stem.split('__')[0]
        piece_id = piece_path.stem.split('__')[1]
        snippet_splits = _assign_splits_by_unperturbed_snippet(piece_id, len(snip_waveforms), test_split, seed)
        # For each of the snippets we should perturb it and save to psnippets_dir
        for i in range(len(snip_waveforms)):
            # Then we should immediately perturb the snippets (this function saves files)
            perturb_snippet(psnippets_new1_dir, piece_path, snip_waveforms[i], i)
            # For each snippet, create 25 rows (1 per perturb)
            for p in range(len(perturbs)):
                pmanifest_rows.append({
                    "filename": f"{composer}__{piece_id}__{perturbs[p]}__s{i:04d}.wav",
                    "composer": composer,
                    "piece_id": piece_id,
                    "perturbation": perturbs[p],
                    "label": f"{composer}__{piece_id}",
                    "snippet_index": i,
                    "split": snippet_splits[i]
                })
        kept_pieces += 1
    
    PMANIFEST_PATH = psnippets_new1_dir / "manifest.csv"
    PMANIFEST_COLUMNS = ["filename", "composer", "piece_id", "perturbation", "label", "snippet_index", "split"]
    with open(PMANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PMANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(pmanifest_rows)

    # 4. Summary
    n_train = sum(1 for r in pmanifest_rows if r["split"] == "train")
    n_test  = sum(1 for r in pmanifest_rows if r["split"] == "test")
    n_labels = len(set(r["label"] for r in pmanifest_rows))

    log.info("=" * 50)
    log.info(f"Pieces kept:    {kept_pieces}  |  skipped: {len(skipped_pieces)}")
    for i in range(len(skipped_pieces)):
        log.info(f"Skipping: {skipped_pieces[i]}")
    log.info(f"Total snippets: {len(pmanifest_rows)}  (train={n_train}, test={n_test})")
    log.info(f"Unique labels:  {n_labels}")
    log.info(f"Snippets saved to: {psnippets_new1_dir}")
    log.info(f"Manifest saved to: {PMANIFEST_PATH}")
    log.info("=" * 50)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    success = run()
    print(success)

#----------------------------------------------------------------------------
# Old code for parts 3, 4, 5, 6 which did perturbations before snippeting
'''
# 3. Load WAV as list[np.ndarray] in waveforms, create tempo/pitch perturbations
    waveforms = [] # all waveforms stored here for perturbing
    valid_wavs = [] # the wav paths of these waveforms
    failed_wf = []
    wavs = load_wav_files(wav_dir)
    for w in wavs:
        wav_path = w['wav_path']
        waveform = embed.load_wav(wav_path)
        if waveform is None:
            failed_wf.append(wav_path.stem)
            continue
        waveforms.append(waveform)
        valid_wavs.append(wav_path)
    if not waveforms:
        log.warning("No waveforms")
        return
    # Perturb all the waveforms
    pwav_dir = perturb_dir / "pwav"
    pwav_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(valid_wavs)):
        perturb(pwav_dir, valid_wavs[i], waveforms[i])
    log.info(f"{len(failed_wf)} failed, {len(valid_wavs)} valid")
    

    # 4. Create snippets with overlap
    # While we technically have the snippets from the non-perturbed/non-overlap,
    # those are from the midi and I don't think we can obtain the exact same by
    # chopping the WAV since there are no exact bar lines.
    folder = Path(pwav_dir)
    pmanifest_rows = []
    kept_pieces = skipped_pieces = 0
    psnippets_dir = perturb_dir / "psnippets"
    psnippets_dir.mkdir(parents=True, exist_ok=True)
    for piece_path in folder.iterdir():
        # Get the composer, piece_id, perturbation by parsing the filename
        filename = piece_path.stem.split("__")
        composer = filename[0]
        piece_id = filename[1]
        perturbation = filename[2] # In form t[float]p[int]
        
        snippets = chop_piece_wav(piece_path, 8, 4)
        if len(snippets) < min_snippets:
            log.debug(f"Skipping {piece_id}: only {len(snippets)} snippets (min={min_snippets}).")
            skipped_pieces += 1
            continue

        # Determine train/test splits
        snippet_splits = data_prep._assign_splits_by_snippet(piece_id, len(snippets), test_split, seed)
        for idx, (snippet, split) in enumerate(zip(snippets, snippet_splits)):
            filename = f"{composer}__{piece_id}__{perturbation}__s{idx:04d}.wav"
            out_path = psnippets_dir / filename
            sf.write(out_path, snippet, SAMPLE_RATE, format='WAV')
            pmanifest_rows.append({
                "filename": filename,
                "composer": composer,
                "piece_id": piece_id,
                "perturbation": perturbation,
                "label": f"{composer}__{piece_id}",
                "snippet_index": idx,
                "split": split
            })
        kept_pieces += 1

    # 5. Write manifest CSV
    PMANIFEST_PATH = psnippets_dir / "manifest.csv"
    PMANIFEST_COLUMNS = ["filename", "composer", "piece_id", "perturbation", "label", "snippet_index", "split"]
    with open(PMANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PMANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(pmanifest_rows)

    # 6. Summary
    n_train = sum(1 for r in pmanifest_rows if r["split"] == "train")
    n_test  = sum(1 for r in pmanifest_rows if r["split"] == "test")
    n_labels = len(set(r["label"] for r in pmanifest_rows))

    log.info("=" * 50)
    log.info(f"Pieces kept:    {kept_pieces}  |  skipped: {skipped_pieces}")
    log.info(f"Total snippets: {len(pmanifest_rows)}  (train={n_train}, test={n_test})")
    log.info(f"Unique labels:  {n_labels}")
    log.info(f"Snippets saved to: {psnippets_dir}")
    log.info(f"Manifest saved to: {PMANIFEST_PATH}")
    log.info("=" * 50)

    return PMANIFEST_PATH
'''