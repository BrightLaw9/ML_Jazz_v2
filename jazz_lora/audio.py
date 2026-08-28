from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .captions import choose_caption
from .config import DataConfig


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}


def discover_audio(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _split_for_block(source: Path, block_index: int, config: DataConfig) -> str:
    key = f"{config.seed}|{source.name}|{block_index}|split".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "holdout" if value < config.holdout_fraction else "train"


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        waveform, target_rate // divisor, source_rate // divisor
    ).astype(np.float32, copy=False)


def _mono(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)


def prepare_clips(
    config: DataConfig,
    *,
    overwrite: bool = False,
    max_clips: int | None = None,
) -> Iterator[dict[str, Any]]:
    sources = discover_audio(config.raw_dir)
    if not sources:
        raise FileNotFoundError(f"No supported audio files found under {config.raw_dir}")

    output_root = Path(config.clip_dir)
    target_frames = round(config.clip_seconds * config.sample_rate)
    block_clips = max(1, round(config.holdout_block_seconds / config.clip_seconds))
    emitted = 0

    for source in sources:
        with sf.SoundFile(source) as audio_file:
            source_rate = audio_file.samplerate
            source_clip_frames = round(config.clip_seconds * source_rate)
            clip_index = 0
            while True:
                block = audio_file.read(
                    frames=source_clip_frames, dtype="float32", always_2d=True
                )
                if len(block) < source_clip_frames:
                    break
                waveform = _resample(_mono(block), source_rate, config.sample_rate)
                if len(waveform) < target_frames:
                    waveform = np.pad(waveform, (0, target_frames - len(waveform)))
                waveform = waveform[:target_frames]

                split = _split_for_block(source, clip_index // block_clips, config)
                source_id = hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:10]
                name = f"{source.stem[:48]}_{source_id}_{clip_index:06d}.wav"
                destination = output_root / split / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not destination.exists():
                    sf.write(destination, waveform, config.sample_rate, subtype="PCM_16")

                yield {
                    "audio_path": destination.as_posix(),
                    "caption": choose_caption(source, clip_index, config.seed),
                    "clip_index": clip_index,
                    "duration_seconds": config.clip_seconds,
                    "source": source.as_posix(),
                    "source_start_seconds": clip_index * config.clip_seconds,
                    "split": split,
                }
                emitted += 1
                clip_index += 1
                if max_clips is not None and emitted >= max_clips:
                    return
