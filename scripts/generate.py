from __future__ import annotations

import argparse

import torch

from jazz_lora.config import load_config
from jazz_lora.inference import generate_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with the base model or a LoRA adapter.")
    parser.add_argument("--config", default="configs/cpu_smoke.json")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="output/sample.wav")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--negative-prompt", default="low quality, noisy, distorted")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    output = generate_one(
        load_config(args.config),
        prompt=args.prompt,
        destination=args.output,
        device=device,
        adapter=args.adapter,
        seed=args.seed,
        inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )
    print(output)


if __name__ == "__main__":
    main()
