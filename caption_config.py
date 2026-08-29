"""User-editable training data categories and captions."""

# Keys are case-insensitive names of immediate subfolders under training_samples.
# Add or replace entries here to train on a different style or instrumentation.
CAPTIONS_BY_FOLDER: dict[str, tuple[str, ...]] = {
    "jazz_piano_trio": (
        "A beautiful jazz music made by piano.",
        "A beautiful jazz piano melody.",
        "Jazz music with a melodic piano.",
    ),
    "jazz_sax": (
        "A beautiful jazz music with saxophone.",
        "A beautiful jazz saxophone melody.",
        "Jazz music with a melodic saxophone.",
    ),
}

# Files in an unmapped folder draw from every configured caption tuple.
DEFAULT_CAPTIONS: tuple[str, ...] = tuple(
    caption
    for captions in CAPTIONS_BY_FOLDER.values()
    for caption in captions
)

if not CAPTIONS_BY_FOLDER or any(
    not captions for captions in CAPTIONS_BY_FOLDER.values()
):
    raise ValueError(
        "CAPTIONS_BY_FOLDER must contain at least one non-empty caption tuple"
    )
