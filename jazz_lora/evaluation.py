from __future__ import annotations

import json
import random
from pathlib import Path

import scipy.io.wavfile
import torch

from .captions import GENERIC_CAPTIONS
from .config import AppConfig
from .inference import load_pipeline


def _generate_condition(
    config: AppConfig,
    destination: Path,
    prompts: list[str],
    seeds: list[int],
    *,
    device: str,
    adapter: str | None,
    inference_steps: int,
) -> None:
    pipe = load_pipeline(config, device, adapter)
    destination.mkdir(parents=True, exist_ok=True)
    for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            audio = pipe(
                prompt=prompt,
                negative_prompt="low quality, noisy, distorted",
                num_inference_steps=inference_steps,
                audio_length_in_s=config.data.clip_seconds,
                generator=generator,
            ).audios[0]
        scipy.io.wavfile.write(
            destination / f"{index:04d}.wav", pipe.vocoder.config.sampling_rate, audio
        )
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()


def evaluate(
    config: AppConfig,
    *,
    adapter: str,
    device: str,
    count: int,
    inference_steps: int,
    compute_fad: bool,
) -> dict[str, float | int | str]:
    rng = random.Random(config.train.seed)
    prompts = [rng.choice(GENERIC_CAPTIONS) for _ in range(count)]
    seeds = [rng.randrange(2**31) for _ in range(count)]
    run_name = f"{Path(adapter).name}_n{count}_s{inference_steps}"
    root = Path(config.train.output_dir) / "eval" / run_name
    _generate_condition(
        config,
        root / "lora_samples",
        prompts,
        seeds,
        device=device,
        adapter=adapter,
        inference_steps=inference_steps,
    )
    _generate_condition(
        config,
        root / "base_samples",
        prompts,
        seeds,
        device=device,
        adapter=None,
        inference_steps=inference_steps,
    )
    result: dict[str, float | int | str] = {
        "adapter": adapter,
        "count_per_condition": count,
        "reference": str(Path(config.data.clip_dir) / "holdout"),
    }
    if compute_fad:
        from frechet_audio_distance import FrechetAudioDistance

        scorer = FrechetAudioDistance(model_name="clap", sample_rate=config.data.sample_rate)
        reference = str(Path(config.data.clip_dir) / "holdout")
        result["fad_lora"] = float(scorer.score(reference, str(root / "lora_samples")))
        result["fad_base"] = float(scorer.score(reference, str(root / "base_samples")))
    (root / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (root / "prompts.json").write_text(
        json.dumps(
            [{"index": i, "prompt": prompt, "seed": seed} for i, (prompt, seed) in enumerate(zip(prompts, seeds))],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
