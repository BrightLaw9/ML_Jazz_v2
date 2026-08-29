from __future__ import annotations

import hashlib
from pathlib import Path

from .config import CAPTIONS_BY_FOLDER, DEFAULT_CAPTIONS


def caption_family(source_path: str | Path) -> tuple[str, ...]:
    category = Path(source_path).parent.name.casefold()
    return next(
        (
            captions
            for folder, captions in CAPTIONS_BY_FOLDER.items()
            if folder.casefold() == category
        ),
        DEFAULT_CAPTIONS,
    )


def choose_caption(source_path: str | Path, clip_index: int, seed: int) -> str:
    source_path = Path(source_path)
    captions = caption_family(source_path)
    digest = hashlib.sha256(
        f"{seed}|{source_path.as_posix()}|{clip_index}|caption".encode("utf-8")
    ).digest()
    return captions[int.from_bytes(digest[:8], "big") % len(captions)]
