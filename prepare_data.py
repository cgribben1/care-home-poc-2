"""Download public datasets and build augmented training clips (no room recordings needed)."""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import ssl
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

# Disable SSL certificate verification — required on Zscaler corporate proxies
# that intercept HTTPS and present a self-signed certificate chain.
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
ssl._create_default_https_context = ssl._create_unverified_context

# Also disable for requests/urllib3 (used by kaggle, yt-dlp).
# Monkey-patch must happen before kaggle is imported anywhere.
try:
    import urllib3
    import requests
    urllib3.disable_warnings()
    _orig_session_send = requests.Session.send

    def _session_send_no_verify(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_session_send(self, request, **kwargs)

    requests.Session.send = _session_send_no_verify
except ImportError:
    pass

import librosa
import numpy as np
import soundfile as sf

from augment import augment_clip
from config import SAMPLE_RATE

COUGHVID_ZENODO_RECORD = "7024894"
COUGHVID_DIR_NAME = "coughvid"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

# Augmentation multipliers per class — room clips get heavy oversampling
AUG_PER_FILE_OVERRIDES: dict[str, int] = {
    "normal_room": 20,  # room ambient clips oversampled heavily
}

ESC50_CATEGORY_MAP = {
    "fall": set(),  # fall corpus = SAFE dataset only
    "cough": {"coughing"},
    "normal": {
        "breathing",
        "footsteps",
        "clock_tick",
        "clock_alarm",
        "washing_machine",
        "vacuum_cleaner",
        "rain",
        "wind",
        "crackling_fire",
        "chirping_birds",
        "snoring",
        "drinking_sipping",
        "brushing_teeth",
        "mouse_click",
        "keyboard_typing",
        "helicopter",
        "engine",
        "train",
        "car_horn",
        "church_bells",
    },
    # Everything else in ESC-50 goes to other
    "other": {
        "glass_breaking",
        "door_wood_knock",
        "door_wood_creaks",
        "can_opening",
        "crying_baby",
        "sneezing",
        "laughing",
        "clapping",
        "fireworks",
        "thunderstorm",
        "rooster",
        "insects",
        "frog",
        "cat",
        "hen",
        "sheep",
        "crow",
        "pig",
        "cow",
        "dog",
        "sea_waves",
        "water_drops",
        "pouring_water",
        "toilet_flush",
        "hand_saw",
        "chainsaw",
        "siren",
        "airplane",
    },
}

# AudioSet class IDs to download for the other class
# Format: {youtube_id_prefix: (audioset_class_id, display_name)}
AUDIOSET_OTHER_CLASSES: dict[str, str] = {
    "/m/0ytgt": "Laughter",
    "/m/09x0r": "Speech",
    "/m/04brg2": "Music",
    "/m/01hgjl": "Television",
    "/m/0dwtp":  "Door",
    "/m/07p6fkf": "Telephone bell ringing",
    "/m/07pp_mv": "Clapping",
    "/m/01j3sz":  "Applause",
    "/m/07pbtc8": "Footsteps, footfall",
    "/m/0jbk":    "Animal",
}
AUDIOSET_BALANCED_CSV_URL = "https://storage.googleapis.com/us_audioset/youtube_enc/v1/balanced_train_segments.csv"
AUDIOSET_MAX_PER_CLASS = 60


def download_esc50(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    esc_root = dest / "ESC-50-master"
    if (esc_root / "audio").exists():
        print(f"ESC-50 already present at {esc_root}")
        return esc_root

    zip_path = dest / "esc50.zip"
    print(f"Downloading ESC-50 from {ESC50_URL} ...")
    with urlopen(ESC50_URL, timeout=120) as resp:
        zip_path.write_bytes(resp.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)
    print(f"ESC-50 extracted to {esc_root}")
    return esc_root


def generate_synthetic_falls(out_dir: Path, count: int = 180, sr: int = SAMPLE_RATE) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(count):
        duration = float(rng.uniform(0.25, 0.9))
        n = int(duration * sr)
        t = np.arange(n, dtype=np.float32) / sr

        decay = float(rng.uniform(6.0, 16.0))
        thump = np.sin(2 * np.pi * float(rng.uniform(80, 220)) * t) * np.exp(-decay * t)
        thump += 0.35 * np.sin(2 * np.pi * float(rng.uniform(220, 480)) * t) * np.exp(
            -(decay * 1.4) * t
        )
        noise = rng.normal(0.0, 0.04, size=n).astype(np.float32)
        signal = (thump + noise).astype(np.float32)
        peak = float(np.max(np.abs(signal))) or 1.0
        signal = signal / peak * float(rng.uniform(0.5, 0.95))

        sf.write(out_dir / f"synthetic_fall_{i:04d}.wav", signal, sr)

    print(f"Generated {count} synthetic fall clips in {out_dir}")


def _load_background_pool(esc_root: Path) -> list[np.ndarray]:
    audio_dir = esc_root / "audio"
    meta_path = esc_root / "meta" / "esc50.csv"
    backgrounds: list[np.ndarray] = []
    with meta_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in ESC50_CATEGORY_MAP["normal"]:
                wav, _ = librosa.load(audio_dir / row["filename"], sr=SAMPLE_RATE, mono=True)
                backgrounds.append(wav.astype(np.float32))
    return backgrounds


def _copy_esc50_classes(esc_root: Path, staging: Path) -> None:
    audio_dir = esc_root / "audio"
    meta_path = esc_root / "meta" / "esc50.csv"
    with meta_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            target = None
            for label, esc_classes in ESC50_CATEGORY_MAP.items():
                if category in esc_classes:
                    target = label
                    break
            if target is None:
                continue
            out_dir = staging / target / "esc50"
            out_dir.mkdir(parents=True, exist_ok=True)
            src = audio_dir / row["filename"]
            dst = out_dir / row["filename"]
            if not dst.exists():
                shutil.copy2(src, dst)


def build_augmented_dataset(
    staging: Path,
    out_dir: Path,
    backgrounds: list[np.ndarray],
    aug_per_file: int = 4,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    counter = 0
    for class_dir in sorted(staging.iterdir()):
        if not class_dir.is_dir():
            continue
        staging_label = class_dir.name
        # normal_room → output label normal, but with heavy oversampling
        output_label = "normal" if staging_label == "normal_room" else staging_label
        n_aug = AUG_PER_FILE_OVERRIDES.get(staging_label, aug_per_file)

        (out_dir / output_label).mkdir(parents=True, exist_ok=True)
        wavs = list(class_dir.rglob("*.wav"))
        for wav_path in wavs:
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)
            sf.write(out_dir / output_label / f"{output_label}_{counter:05d}_clean.wav", audio, SAMPLE_RATE)
            counter += 1

            for aug_idx in range(n_aug):
                bg = backgrounds[int(rng.integers(0, len(backgrounds)))] if backgrounds else None
                aug = augment_clip(audio, SAMPLE_RATE, bg, use_rir=True, use_noise=True, use_gain=True)
                sf.write(
                    out_dir / output_label / f"{output_label}_{counter:05d}_aug{aug_idx}.wav",
                    aug,
                    SAMPLE_RATE,
                )
                counter += 1

    print(f"Built augmented dataset at {out_dir} ({counter} clips total)")


def _kaggle_download(dataset: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset via Python API. Returns True on success."""
    try:
        import kaggle  # type: ignore
        dest.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        return True
    except Exception as exc:
        print(f"Kaggle download failed for {dataset}: {exc}")
        return False


def download_safe(raw_dir: Path) -> None:
    """Download SAFE fall audio dataset from Kaggle. Raises on failure — SAFE is required."""
    safe_dir = raw_dir / "safe"
    if safe_dir.exists() and any(safe_dir.rglob("*.wav")):
        print(f"SAFE dataset already present ({len(list(safe_dir.rglob('*.wav')))} clips)")
        return
    print("Downloading SAFE fall dataset from Kaggle...")
    if not _kaggle_download("antonygarciag/fall-audio-detection-dataset", safe_dir):
        raise RuntimeError(
            "SAFE dataset download failed. Check Kaggle credentials in ~/.kaggle/kaggle.json."
        )
    count = len(list(safe_dir.rglob("*.wav")))
    print(f"SAFE dataset ready ({count} clips)")


def try_download_icbhi(raw_dir: Path) -> None:
    """Download ICBHI respiratory sound database from Kaggle."""
    icbhi_dir = raw_dir / "icbhi"
    if icbhi_dir.exists() and any(icbhi_dir.rglob("*.wav")):
        print("ICBHI dataset already present")
        return
    print("Downloading ICBHI respiratory sound database from Kaggle...")
    if _kaggle_download("vbookshelf/respiratory-sound-database", icbhi_dir):
        count = len(list(icbhi_dir.rglob("*.wav")))
        print(f"ICBHI dataset ready ({count} clips)")
    else:
        print("ICBHI download skipped")


def import_icbhi_clips(staging: Path, raw_dir: Path) -> None:
    """
    Import ICBHI respiratory clips into the cough class.
    ICBHI labels: crackle, wheeze, both, none — we take crackle/wheeze/both as cough-adjacent.
    """
    icbhi_dir = raw_dir / "icbhi"
    if not icbhi_dir.exists():
        return

    # ICBHI filenames encode diagnosis: <patient>_<session>_<loc>_<mode>_<equipment>.wav
    # Annotation files (.txt) have columns: start end crackle wheeze
    annotation_files = list(icbhi_dir.rglob("*.txt"))
    if not annotation_files:
        print("ICBHI: no annotation files found — skipping")
        return

    out = staging / "cough" / "icbhi"
    out.mkdir(parents=True, exist_ok=True)
    copied = 0

    for ann_path in annotation_files:
        wav_path = ann_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        try:
            with open(ann_path) as f:
                rows = [line.strip().split() for line in f if line.strip()]
            # Keep files that have at least one adventitious sound (crackle or wheeze)
            has_event = any(
                len(r) >= 4 and (r[2] == "1" or r[3] == "1") for r in rows
            )
            if has_event:
                dst = out / wav_path.name
                if not dst.exists():
                    shutil.copy2(wav_path, dst)
                copied += 1
        except Exception:
            continue

    print(f"Imported {copied} ICBHI respiratory clips -> staging/cough/icbhi/")


def import_safe_falls(staging: Path, raw_dir: Path) -> None:
    safe_dir = raw_dir / "safe"
    if not safe_dir.exists():
        return
    out = staging / "fall" / "safe"
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    # Every file in the SAFE dataset is a fall — filenames don't contain "fall"
    for wav in safe_dir.rglob("*.wav"):
        dst = out / f"safe_{copied:04d}.wav"
        if not dst.exists():
            shutil.copy2(wav, dst)
        copied += 1
    if copied:
        print(f"Imported {copied} SAFE fall clips")


def _get_ffmpeg() -> str | None:
    """Return path to ffmpeg binary — bundled via imageio-ffmpeg if available."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _webm_to_wav(src: Path, dst: Path, ffmpeg: str) -> bool:
    """Convert a WebM/OGG audio file to 16kHz mono WAV. Returns True on success."""
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def download_coughvid(raw_dir: Path, max_clips: int = 12000) -> Path | None:
    """
    Download COUGHVID from Zenodo, filter to quality != poor, convert to WAV.
    Returns the output directory path, or None if download fails.
    """
    out_dir = raw_dir / COUGHVID_DIR_NAME
    done_marker = out_dir / ".done"
    if done_marker.exists():
        existing = list(out_dir.glob("*.wav"))
        print(f"COUGHVID already present ({len(existing)} WAV clips) at {out_dir}")
        return out_dir

    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        print("COUGHVID skipped: ffmpeg not available (run: pip install imageio-ffmpeg)")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = "public_dataset_v3.zip"
    zip_path = out_dir / zip_filename

    # If the zip was downloaded manually, skip the network fetch
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        print(f"COUGHVID zip found at {zip_path} — skipping download")
    else:
        # Resolve download URL from Zenodo API (handles old list and new dict formats)
        api_url = f"https://zenodo.org/api/records/{COUGHVID_ZENODO_RECORD}"
        print("Fetching COUGHVID record metadata from Zenodo...")
        zip_url: str | None = None
        try:
            with urlopen(api_url, timeout=30) as resp:
                import json
                record = json.loads(resp.read())
            raw_files = record.get("files", [])
            if isinstance(raw_files, list):
                file_map = {f["key"]: f["links"]["self"] for f in raw_files}
            elif isinstance(raw_files, dict):
                file_map = {
                    k: (v["links"]["self"] if isinstance(v, dict) and "links" in v else v)
                    for k, v in raw_files.items()
                    if k != "enabled"
                }
            else:
                file_map = {}
            zip_key = next((k for k in file_map if k.endswith(".zip")), None)
            if zip_key:
                zip_url = file_map[zip_key]
                zip_filename = zip_key
                zip_path = out_dir / zip_filename
        except Exception as exc:
            print(f"  Zenodo API unavailable ({exc}) — falling back to direct URL")

        if not zip_url:
            zip_url = f"https://zenodo.org/record/{COUGHVID_ZENODO_RECORD}/files/{zip_filename}"

        print(f"Downloading COUGHVID zip ({zip_filename}) — this may take a while...")
        try:
            with urlopen(zip_url, timeout=1200) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {pct:.0f}%  ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="", flush=True)
            print()
        except Exception as exc:
            print(f"\nCOUGHVID skipped: download failed — {exc}")
            return None

    # Read metadata CSV from inside the zip to filter by quality
    good_uuids: set[str] | None = None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if csv_name:
                with zf.open(csv_name) as cf:
                    meta_csv = cf.read().decode("utf-8", errors="replace")
                rows = list(csv.DictReader(io.StringIO(meta_csv)))
                good_rows = [r for r in rows if r.get("quality", "").strip().lower() != "poor"]
                good_uuids = {r.get("uuid", "").strip() for r in good_rows}
                print(f"COUGHVID: {len(rows)} clips, {len(good_rows)} after quality filter")
            else:
                print("COUGHVID: no metadata CSV in zip — taking all clips")
    except Exception as exc:
        print(f"COUGHVID: could not read metadata CSV ({exc}) — taking all clips")

    webm_dir = out_dir / "webm"
    webm_dir.mkdir(exist_ok=True)

    print(f"Extracting and converting up to {max_clips} clips...")
    converted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            if converted >= max_clips:
                break
            stem = Path(name).stem
            if name.lower().endswith(".csv"):
                continue
            if good_uuids is not None and stem not in good_uuids:
                continue
            wav_path = out_dir / f"{stem}.wav"
            if wav_path.exists():
                converted += 1
                continue
            try:
                zf.extract(name, webm_dir)
                extracted = webm_dir / name
                if _webm_to_wav(extracted, wav_path, ffmpeg):
                    converted += 1
                    extracted.unlink(missing_ok=True)
                else:
                    skipped += 1
                    extracted.unlink(missing_ok=True)
            except Exception:
                skipped += 1
            if converted % 500 == 0 and converted:
                print(f"  converted {converted} clips...")

    zip_path.unlink(missing_ok=True)
    shutil.rmtree(webm_dir, ignore_errors=True)
    done_marker.touch()
    print(f"COUGHVID: {converted} WAV clips saved to {out_dir} ({skipped} skipped)")
    return out_dir


def import_coughvid_clips(staging: Path, raw_dir: Path) -> None:
    coughvid_dir = raw_dir / COUGHVID_DIR_NAME
    if not coughvid_dir.exists():
        return
    wavs = list(coughvid_dir.glob("*.wav"))
    if not wavs:
        return
    out = staging / "cough" / "coughvid"
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wav in wavs:
        dst = out / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
        copied += 1
    print(f"Imported {copied} COUGHVID clips -> staging/cough/coughvid/")


def import_room_clips(staging: Path, cal_dir: Path, chunk_sec: float = 5.0) -> None:
    """Split calibration room recordings into chunks → staging/normal_room/ (×20 aug)."""
    room_wavs = [f for f in cal_dir.glob("room_ambient_*.wav") if "latest" not in f.name]
    if not room_wavs:
        print("No room ambient recordings in data/calibration/ — skipping room clips")
        return
    out = staging / "normal_room" / "room"
    out.mkdir(parents=True, exist_ok=True)
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    total = 0
    for wav_path in sorted(room_wavs):
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        n_chunks = len(audio) // chunk_samples
        for i in range(n_chunks):
            chunk_audio = audio[i * chunk_samples : (i + 1) * chunk_samples]
            dst = out / f"{wav_path.stem}_chunk{i:03d}.wav"
            if not dst.exists():
                sf.write(dst, chunk_audio, SAMPLE_RATE)
            total += 1
    print(f"Staged {total} room ambient chunks -> staging/normal_room/ (×20 aug)")


def _ytdlp_bin() -> str | None:
    """Return path to yt-dlp — venv Scripts first, then system PATH."""
    for candidate in [
        Path(__file__).resolve().parent / "venv" / "Scripts" / "yt-dlp.exe",
        Path(__file__).resolve().parent / "venv" / "Scripts" / "yt-dlp",
    ]:
        if candidate.exists():
            return str(candidate)
    return shutil.which("yt-dlp")


def download_audioset_other(raw_dir: Path, max_per_class: int = AUDIOSET_MAX_PER_CLASS) -> Path | None:
    """Download AudioSet clips for the 'other' class via yt-dlp."""
    out_dir = raw_dir / "audioset_other"
    done_marker = out_dir / ".done"
    if done_marker.exists():
        existing = list(out_dir.glob("*.wav"))
        print(f"AudioSet other already present ({len(existing)} clips)")
        return out_dir

    ytdlp = _ytdlp_bin()
    if not ytdlp:
        print("AudioSet skipped: yt-dlp not found (run: pip install yt-dlp)")
        return None

    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        print("AudioSet skipped: ffmpeg not available")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Download or reuse the AudioSet balanced segments CSV
    csv_path = raw_dir / "balanced_train_segments.csv"
    if not csv_path.exists():
        print("Downloading AudioSet balanced_train_segments.csv...")
        try:
            with urlopen(AUDIOSET_BALANCED_CSV_URL, timeout=120) as resp:
                csv_path.write_bytes(resp.read())
        except Exception as exc:
            print(f"AudioSet skipped: CSV download failed — {exc}")
            return None

    # Parse CSV — format: # YTID, start_seconds, end_seconds, positive_labels
    target_ids = set(AUDIOSET_OTHER_CLASSES.keys())
    clips_by_class: dict[str, list[tuple[str, float, float]]] = {k: [] for k in target_ids}
    with csv_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(", ", 3)
            if len(parts) < 4:
                continue
            ytid, start_str, end_str, labels_str = parts
            labels = {lbl.strip().strip('"') for lbl in labels_str.split(",")}
            for class_id in labels & target_ids:
                if len(clips_by_class[class_id]) < max_per_class:
                    clips_by_class[class_id].append((ytid, float(start_str), float(end_str)))

    total_downloaded = 0
    for class_id, clips in clips_by_class.items():
        class_name = AUDIOSET_OTHER_CLASSES[class_id].replace(" ", "_").replace(",", "")
        print(f"AudioSet '{class_name}': attempting {len(clips)} clips...")
        class_ok = 0
        for ytid, start, end in clips:
            out_wav = out_dir / f"audioset_{class_name}_{ytid}.wav"
            if out_wav.exists():
                class_ok += 1
                continue
            url = f"https://www.youtube.com/watch?v={ytid}"
            tmp_base = out_dir / f"_tmp_{ytid}"
            try:
                subprocess.run(
                    [
                        ytdlp, "-x", "--audio-format", "wav",
                        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
                        "--download-sections", f"*{start}-{end}",
                        "--no-playlist", "-q",
                        "-o", str(tmp_base) + ".%(ext)s",
                        url,
                    ],
                    capture_output=True,
                    timeout=60,
                )
                candidates = list(out_dir.glob(f"_tmp_{ytid}*"))
                if not candidates:
                    continue
                best = candidates[0]
                if best.suffix.lower() == ".wav":
                    best.rename(out_wav)
                    class_ok += 1
                    total_downloaded += 1
                else:
                    r = subprocess.run(
                        [ffmpeg, "-y", "-i", str(best), "-ar", "16000", "-ac", "1", str(out_wav)],
                        capture_output=True, timeout=30,
                    )
                    best.unlink(missing_ok=True)
                    if r.returncode == 0 and out_wav.exists():
                        class_ok += 1
                        total_downloaded += 1
            except Exception:
                pass
            finally:
                for tmp in out_dir.glob(f"_tmp_{ytid}*"):
                    tmp.unlink(missing_ok=True)
        print(f"  {class_ok}/{len(clips)} downloaded for '{class_name}'")

    done_marker.touch()
    print(f"AudioSet other: {total_downloaded} total clips saved to {out_dir}")
    return out_dir


def import_coughvid_from_extracted(
    src_dir: Path,
    raw_dir: Path,
    max_clips: int = 12000,
    cough_threshold: float = 0.5,
) -> None:
    """
    Convert already-extracted COUGHVID webm/wav files to 16kHz WAV in data/raw/coughvid/.
    Filters by cough_detected >= cough_threshold from each clip's JSON sidecar.
    Skips if data/raw/coughvid/.done already exists.
    """
    out_dir = raw_dir / COUGHVID_DIR_NAME
    done_marker = out_dir / ".done"
    if done_marker.exists():
        existing = list(out_dir.glob("*.wav"))
        print(f"COUGHVID already present ({len(existing)} WAV clips) — skipping import")
        return

    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        print("COUGHVID skipped: ffmpeg not available (pip install imageio-ffmpeg)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidates — prefer webm (higher count), supplement with wav
    import json as _json
    candidates = sorted(src_dir.glob("*.webm")) + sorted(src_dir.glob("*.wav"))

    converted = 0
    skipped = 0
    for src in candidates:
        if converted >= max_clips:
            break
        uuid = src.stem
        dst = out_dir / f"{uuid}.wav"
        if dst.exists():
            converted += 1
            continue

        # Quality filter via JSON sidecar
        json_path = src.with_suffix(".json")
        if json_path.exists():
            try:
                meta = _json.loads(json_path.read_text(encoding="utf-8"))
                score = float(meta.get("cough_detected", 0))
                if score < cough_threshold:
                    skipped += 1
                    continue
            except Exception:
                pass  # no JSON or parse error → keep clip

        if src.suffix.lower() == ".wav":
            # Already WAV — just resample to 16kHz mono
            try:
                audio, sr = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
                sf.write(dst, audio.astype(np.float32), SAMPLE_RATE)
                converted += 1
            except Exception:
                skipped += 1
        else:
            if _webm_to_wav(src, dst, ffmpeg):
                converted += 1
            else:
                skipped += 1

        if converted % 500 == 0 and converted:
            print(f"  COUGHVID: converted {converted} clips...")

    done_marker.touch()
    print(f"COUGHVID: {converted} clips saved to {out_dir} ({skipped} skipped/below threshold)")


def import_audioset_other(staging: Path, raw_dir: Path) -> None:
    audioset_dir = raw_dir / "audioset_other"
    if not audioset_dir.exists():
        return
    wavs = list(audioset_dir.glob("*.wav"))
    if not wavs:
        return
    out = staging / "other" / "audioset"
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wav in wavs:
        dst = out / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
        copied += 1
    print(f"Imported {copied} AudioSet other clips -> staging/other/audioset/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare training data from public sources")
    parser.add_argument("--aug-per-file", type=int, default=4)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    staging = DATA_DIR / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    cal_dir = ROOT / "data" / "calibration"

    esc_root = download_esc50(RAW_DIR / "esc50")
    _copy_esc50_classes(esc_root, staging)
    download_safe(RAW_DIR)
    import_safe_falls(staging, RAW_DIR)
    coughvid_extracted = Path(r"C:\Users\cugribb\Downloads\public_dataset_v3\coughvid_20211012")
    import_coughvid_from_extracted(coughvid_extracted, RAW_DIR)
    import_coughvid_clips(staging, RAW_DIR)
    # download_audioset_other(RAW_DIR)    # uncomment once balanced_train_segments.csv is in data/raw/
    # import_audioset_other(staging, RAW_DIR)
    import_room_clips(staging, cal_dir)

    backgrounds = _load_background_pool(esc_root)
    build_augmented_dataset(staging, TRAIN_DIR, backgrounds, aug_per_file=args.aug_per_file)
    print("\nData ready. Next: python train_classifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
