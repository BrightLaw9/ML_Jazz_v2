from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


# Keys are case-insensitive names of immediate subfolders under training_samples.
# Add or replace entries here to train on a different style or instrumentation.
CAPTIONS_BY_FOLDER: dict[str, tuple[str, ...]] = {
    "jazz_piano_trio": (
        "A beautiful jazz music made by piano.",
        "A beautiful jazz piano melody.",
        "Jazz music with a melodic piano.",
    ),
    "jazz_sax": (
        "A beautiful jazz music with saxophone.",
        "A beautiful jazz saxophone melody.",
        "Jazz music with a melodic saxophone.",
    ),
}

# Files in an unmapped folder draw from every configured caption tuple.
DEFAULT_CAPTIONS: tuple[str, ...] = tuple(
    caption
    for captions in CAPTIONS_BY_FOLDER.values()
    for caption in captions
)

if not CAPTIONS_BY_FOLDER or any(
    not captions for captions in CAPTIONS_BY_FOLDER.values()
):
    raise ValueError(
        "CAPTIONS_BY_FOLDER must contain at least one non-empty caption tuple"
    )


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "training_samples"
    clip_dir: str = "data/clips"
    manifest_path: str = "data/manifest.jsonl"
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    holdout_fraction: float = 0.1
    holdout_block_seconds: float = 60.0
    seed: int = 42


@dataclass(frozen=True)
class MelConfig:
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 8000.0
    log_clip: float = 1e-5


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "cvssp/audioldm2-music"
    revision: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "output"
    latent_dir: str = "data/latents"
    latent_manifest_path: str = "data/latent_manifest.jsonl"
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    max_steps: int | None = None
    max_train_samples: int | None = None
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    empty_prompt_probability: float = 0.1
    mixed_precision: str = "no"
    gradient_checkpointing: bool = False
    num_workers: int = 0
    checkpoint_every_steps: int = 500
    sample_on_checkpoint: bool = False
    sample_inference_steps: int = 20
    sample_prompt: str = (
        "A beautiful jazz piano melody."
    )
    seed: int = 42
    init_adapter_from: str | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    mel: MelConfig
    model: ModelConfig
    train: TrainConfig


def _build(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    if cls is ModelConfig and "target_modules" in values:
        values = {**values, "target_modules": tuple(values["target_modules"])}
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    expected = {"data", "mel", "model", "train"}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    config = AppConfig(
        data=_build(DataConfig, raw.get("data", {})),
        mel=_build(MelConfig, raw.get("mel", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.data.sample_rate <= 0 or config.data.clip_seconds <= 0:
        raise ValueError("sample_rate and clip_seconds must be positive")
    if not 0 <= config.data.holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if config.mel.f_max > config.data.sample_rate / 2:
        raise ValueError("mel.f_max cannot exceed the Nyquist frequency")
    if config.model.lora_rank <= 0 or config.model.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if config.train.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be one of: no, fp16, bf16")
    if not 0 <= config.train.empty_prompt_probability <= 1:
        raise ValueError("empty_prompt_probability must be in [0, 1]")
    if config.train.init_adapter_from and config.train.resume_from:
        raise ValueError(
            "train.init_adapter_from and train.resume_from are mutually exclusive"
        )
