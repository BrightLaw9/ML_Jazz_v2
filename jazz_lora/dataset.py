from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .manifest import read_jsonl


class LatentCaptionDataset(Dataset):
    def __init__(self, manifest_path: str, max_samples: int | None = None) -> None:
        self.rows = list(read_jsonl(manifest_path))
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No cached training examples in {manifest_path}")
        missing = [row["latent_path"] for row in self.rows if not Path(row["latent_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing latent file: {missing[0]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        payload = torch.load(row["latent_path"], map_location="cpu", weights_only=True)
        latent = payload["latent"] if isinstance(payload, dict) else payload
        return latent.float(), row["caption"]
