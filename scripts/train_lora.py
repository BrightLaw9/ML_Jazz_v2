from __future__ import annotations

import argparse

from jazz_lora.config import load_config
from jazz_lora.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AudioLDM2 with a LoRA adapter.")
    parser.add_argument("--config", default="configs/cpu_smoke.json")
    args = parser.parse_args()
    destination = train(load_config(args.config))
    print(f"Final adapter: {destination}")


if __name__ == "__main__":
    main()
