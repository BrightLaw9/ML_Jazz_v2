from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch

from .config import AppConfig
from .manifest import read_jsonl, write_jsonl
from .mel import AudioLDM2Mel


def _load_clip(path: str, expected_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path} has sample rate {sample_rate}; expected {expected_sample_rate}. "
            "Run prepare_data.py rather than pointing the cache at raw recordings."
        )
    return torch.from_numpy(waveform.mean(axis=1))


def cache_latents(
    config: AppConfig,
    *,
    device: str,
    overwrite: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    from diffusers import AutoencoderKL

    torch_device = torch.device(device)
    vae = AutoencoderKL.from_pretrained(
        config.model.model_id,
        subfolder="vae",
        revision=config.model.revision,
        torch_dtype=torch.float32,
    ).to(torch_device)
    vae.requires_grad_(False).eval()
    transform = AudioLDM2Mel(config.data, config.mel).to(torch_device)
    latent_root = Path(config.train.latent_dir)
    latent_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=torch_device).manual_seed(config.train.seed)

    train_rows: Iterable[dict[str, Any]] = (
        row for row in read_jsonl(config.data.manifest_path) if row["split"] == "train"
    )
    for index, row in enumerate(train_rows):
        if max_samples is not None and index >= max_samples:
            break
        audio_path = Path(row["audio_path"])
        latent_path = latent_root / f"{audio_path.stem}.pt"
        if overwrite or not latent_path.exists():
            waveform = _load_clip(str(audio_path), config.data.sample_rate).to(torch_device)
            with torch.inference_mode():
                mel = transform(waveform)
                latent = vae.encode(mel.to(dtype=vae.dtype)).latent_dist.sample(
                    generator=generator
                )
                latent = latent * vae.config.scaling_factor
            payload = {
                "latent": latent.squeeze(0).cpu().float(),
                "mel_shape": tuple(mel.shape),
                "scaling_factor": float(vae.config.scaling_factor),
            }
            torch.save(payload, latent_path)
        rows.append({**row, "latent_path": latent_path.as_posix()})

    write_jsonl(config.train.latent_manifest_path, rows)
    return rows
