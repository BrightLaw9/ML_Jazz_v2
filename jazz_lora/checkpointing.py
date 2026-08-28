from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def adapter_state_dict(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    from peft.utils import get_peft_model_state_dict

    return {key: value.detach().cpu() for key, value in get_peft_model_state_dict(unet).items()}


def save_checkpoint(
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: str | Path,
    *,
    global_step: int,
    epoch: int,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    state = adapter_state_dict(unet)
    torch.save(state, destination / "adapter_state.pt")
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step, "epoch": epoch},
        destination / "training_state.pt",
    )
    from safetensors.torch import save_file

    save_file(
        {key: value.contiguous() for key, value in state.items()},
        destination / "adapter_model.safetensors",
        metadata={"format": "pt", "adapter_name": "default"},
    )
    (destination / "metadata.json").write_text(
        json.dumps({"global_step": global_step, "epoch": epoch}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_adapter_for_inference(
    unet: torch.nn.Module, directory: str | Path, model_config: object
) -> None:
    """Load this project's checkpoint without relying on pipeline-specific LoRA mixins."""
    from peft.utils import set_peft_model_state_dict

    from .modeling import attach_lora

    source = Path(directory)
    raw_state = source / "adapter_state.pt"
    safe_state = source / "adapter_model.safetensors"
    attach_lora(unet, model_config)
    if raw_state.is_file():
        state = torch.load(raw_state, map_location="cpu", weights_only=True)
    elif safe_state.is_file():
        from safetensors.torch import load_file

        state = load_file(safe_state, device="cpu")
    else:
        raise FileNotFoundError(f"No adapter_state.pt or adapter_model.safetensors in {source}")
    result = set_peft_model_state_dict(unet, state)
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def initialize_adapter_weights(
    unet: torch.nn.Module,
    directory: str | Path,
) -> None:
    """Initialize an already-attached LoRA without restoring training progress.

    This is intended for a new fine-tuning phase with a different dataset. The
    caller creates a fresh optimizer and starts its phase-local step counter at
    zero.
    """
    from peft.utils import set_peft_model_state_dict

    source = Path(directory)
    raw_state = source / "adapter_state.pt"
    safe_state = source / "adapter_model.safetensors"
    if raw_state.is_file():
        adapter = torch.load(raw_state, map_location="cpu", weights_only=True)
    elif safe_state.is_file():
        from safetensors.torch import load_file

        adapter = load_file(safe_state, device="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter_state.pt or adapter_model.safetensors in {source}"
        )
    result = set_peft_model_state_dict(unet, adapter)
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def restore_checkpoint(
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: str | Path,
) -> tuple[int, int]:
    from peft.utils import set_peft_model_state_dict

    source = Path(directory)
    adapter = torch.load(source / "adapter_state.pt", map_location="cpu", weights_only=True)
    result = set_peft_model_state_dict(unet, adapter)
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
    state: dict[str, Any] = torch.load(
        source / "training_state.pt", map_location="cpu", weights_only=True
    )
    optimizer.load_state_dict(state["optimizer"])
    return int(state["global_step"]), int(state["epoch"])
