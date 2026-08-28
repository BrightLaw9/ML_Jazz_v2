from __future__ import annotations

import hashlib
from pathlib import Path


TRIO_CAPTIONS = (
    "A beautiful jazz music made by piano.",
    "A beautiful jazz piano melody.",
    "Jazz music with a melodic piano.",
)

SAX_CAPTIONS = (
    "A beautiful jazz music with saxophone.",
    "A beautiful jazz saxophone melody.",
    "Jazz music with a melodic saxophone.",
)

GENERIC_CAPTIONS = TRIO_CAPTIONS + SAX_CAPTIONS


def caption_family(source_path: str | Path) -> tuple[str, ...]:
    category = Path(source_path).parent.name.casefold()
    if category == "jazz_sax":
        return SAX_CAPTIONS
    if category == "jazz_piano_trio":
        return TRIO_CAPTIONS
    return GENERIC_CAPTIONS


def choose_caption(source_path: str | Path, clip_index: int, seed: int) -> str:
    source_path = Path(source_path)
    captions = caption_family(source_path)
    digest = hashlib.sha256(
        f"{seed}|{source_path.as_posix()}|{clip_index}|caption".encode("utf-8")
    ).digest()
    return captions[int.from_bytes(digest[:8], "big") % len(captions)]
