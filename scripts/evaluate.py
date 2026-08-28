from __future__ import annotations

import argparse
import json

import torch

from jazz_lora.config import load_config
from jazz_lora.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched LoRA/base generations and optional FAD.")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fad", action="store_true")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    result = evaluate(
        load_config(args.config),
        adapter=args.adapter,
        device=device,
        count=args.count,
        inference_steps=args.steps,
        compute_fad=args.fad,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
