from __future__ import annotations

import argparse

import torch

from jazz_lora.cache import cache_latents
from jazz_lora.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache AudioLDM2 VAE latents for training clips.")
    parser.add_argument("--config", default="configs/cpu_smoke.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rows = cache_latents(
        config, device=device, overwrite=args.overwrite, max_samples=args.max_samples
    )
    print(f"Cached/verified {len(rows)} latents in {config.train.latent_dir}")


if __name__ == "__main__":
    main()
