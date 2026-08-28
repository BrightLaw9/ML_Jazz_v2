from __future__ import annotations

from types import MethodType

import torch

from .config import ModelConfig


def ensure_language_model_compatibility(pipe: object) -> bool:
    """Support legacy AudioLDM2 checkpoints whose language model is GPT2Model.

    Diffusers 0.39 calls GenerationMixin internals that only GPT2LMHeadModel exposes,
    while cvssp/audioldm2-music's model index loads the original GPT2Model. The
    original AudioLDM2 behavior is an autoregressive loop over hidden states.
    """
    language_model = pipe.language_model
    if hasattr(language_model, "_update_model_kwargs_for_generation"):
        return False

    def generate_language_model(
        self: object,
        inputs_embeds: torch.Tensor | None = None,
        max_new_tokens: int | None = 8,
        **model_kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")
        token_count = (
            max_new_tokens
            if max_new_tokens is not None
            else self.language_model.config.max_new_tokens
        )
        attention_mask = model_kwargs.get("attention_mask")
        generated = inputs_embeds
        for _ in range(token_count):
            output = self.language_model(
                inputs_embeds=generated,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            next_hidden_state = output.hidden_states[-1][:, -1:, :]
            generated = torch.cat((generated, next_hidden_state), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            attention_mask.shape[0],
                            1,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
        return generated[:, -token_count:, :]

    pipe.generate_language_model = MethodType(generate_language_model, pipe)
    return True


def freeze_pipeline(pipe: object) -> None:
    names = (
        "vae",
        "text_encoder",
        "text_encoder_2",
        "projection_model",
        "language_model",
        "vocoder",
        "unet",
    )
    for name in names:
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()


def attach_lora(unet: torch.nn.Module, config: ModelConfig) -> None:
    from peft import LoraConfig, inject_adapter_in_model

    target_modules: str | list[str]
    if len(config.target_modules) == 1 and config.target_modules[0].startswith(".*"):
        target_modules = config.target_modules[0]
    else:
        target_modules = list(config.target_modules)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    injected = inject_adapter_in_model(lora_config, unet, adapter_name="default")
    if injected is not unet:
        raise RuntimeError("PEFT unexpectedly replaced the UNet instead of modifying it in place")


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def parameter_summary(module: torch.nn.Module) -> tuple[int, int, float]:
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in module.parameters())
    return trainable, total, 100.0 * trainable / total


def diffusion_target(scheduler: object, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    if prediction_type == "sample":
        return latents
    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")
