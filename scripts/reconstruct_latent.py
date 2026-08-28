from __future__ import annotations

import argparse

import torch

from jazz_lora.config import load_config
from jazz_lora.inference import reconstruct_latent
from jazz_lora.manifest import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a cached latent to validate preprocessing.")
    parser.add_argument("--config", default="configs/cpu_smoke.json")
    parser.add_argument("--latent")
    parser.add_argument("--output", default="output/reconstruction.wav")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    latent = args.latent or next(read_jsonl(config.train.latent_manifest_path))["latent_path"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(reconstruct_latent(config, latent_path=latent, destination=args.output, device=device))


if __name__ == "__main__":
    main()
