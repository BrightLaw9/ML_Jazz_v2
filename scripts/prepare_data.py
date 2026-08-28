from __future__ import annotations

import argparse
from collections import Counter

from jazz_lora.audio import prepare_clips
from jazz_lora.config import load_config
from jazz_lora.manifest import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream, resample, split, and caption source audio.")
    parser.add_argument("--config", default="configs/cpu_smoke.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = list(
        prepare_clips(config.data, overwrite=args.overwrite, max_clips=args.max_clips)
    )
    write_jsonl(config.data.manifest_path, rows)
    counts = Counter(row["split"] for row in rows)
    print(f"Wrote {len(rows)} clips and {config.data.manifest_path}: {dict(counts)}")


if __name__ == "__main__":
    main()
