from __future__ import annotations

import json
from pathlib import Path

import scipy.io.wavfile
import torch

from .config import AppConfig


def load_pipeline(config: AppConfig, device: str, adapter: str | None = None):
    from diffusers import AudioLDM2Pipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        torch_dtype=dtype,
    ).to(device)
    from .modeling import ensure_language_model_compatibility

    ensure_language_model_compatibility(pipe)
    if adapter:
        from .checkpointing import load_adapter_for_inference

        load_adapter_for_inference(pipe.unet, adapter, config.model)
    return pipe


def generate_one(
    config: AppConfig,
    *,
    prompt: str,
    destination: str | Path,
    device: str,
    adapter: str | None,
    seed: int,
    inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
) -> Path:
    pipe = load_pipeline(config, device, adapter)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            audio_length_in_s=config.data.clip_seconds,
            generator=generator,
        ).audios[0]
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(output, pipe.vocoder.config.sampling_rate, audio)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "adapter": adapter,
                "guidance_scale": guidance_scale,
                "inference_steps": inference_steps,
                "model_id": config.model.model_id,
                "negative_prompt": negative_prompt,
                "prompt": prompt,
                "seed": seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def reconstruct_latent(
    config: AppConfig,
    *,
    latent_path: str | Path,
    destination: str | Path,
    device: str,
) -> Path:
    pipe = load_pipeline(config, device)
    payload = torch.load(latent_path, map_location=device, weights_only=True)
    latent = payload["latent"] if isinstance(payload, dict) else payload
    latent = latent.unsqueeze(0).to(device=device, dtype=pipe.vae.dtype)
    with torch.inference_mode():
        mel = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        waveform = pipe.mel_spectrogram_to_waveform(mel)[0].numpy()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(output, pipe.vocoder.config.sampling_rate, waveform)
    return output
