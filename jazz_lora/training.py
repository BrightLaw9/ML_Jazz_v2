from __future__ import annotations

import csv
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpointing import initialize_adapter_weights, restore_checkpoint, save_checkpoint
from .config import AppConfig
from .dataset import LatentCaptionDataset
from .modeling import (
    attach_lora,
    diffusion_target,
    ensure_language_model_compatibility,
    freeze_pipeline,
    parameter_summary,
    trainable_parameters,
)


def _sample_checkpoint(pipe: object, config: AppConfig, destination: Path) -> None:
    import scipy.io.wavfile

    pipe.unet.eval()
    with torch.inference_mode():
        audio = pipe(
            prompt=config.train.sample_prompt,
            negative_prompt="low quality, noisy, distorted",
            num_inference_steps=config.train.sample_inference_steps,
            audio_length_in_s=config.data.clip_seconds,
        ).audios[0]
    scipy.io.wavfile.write(destination / "sample.wav", config.data.sample_rate, audio)
    pipe.unet.train()


def train(config: AppConfig) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AudioLDM2Pipeline

    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision,
    )
    set_seed(config.train.seed)
    output_root = Path(config.train.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pipe = AudioLDM2Pipeline.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        torch_dtype=torch.float32,
    )
    if ensure_language_model_compatibility(pipe):
        accelerator.print("Enabled GPT2Model compatibility for the AudioLDM2 checkpoint")
    freeze_pipeline(pipe)
    attach_lora(pipe.unet, config.model)
    if config.train.init_adapter_from:
        initialize_adapter_weights(pipe.unet, config.train.init_adapter_from)
        accelerator.print(
            "Initialized LoRA weights for a new training phase from "
            f"{config.train.init_adapter_from}; optimizer and global_step reset"
        )
    if config.train.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    pipe.to(accelerator.device)

    parameters = trainable_parameters(pipe.unet)
    if not parameters:
        raise RuntimeError("LoRA attachment produced no trainable parameters")
    trainable, total, percentage = parameter_summary(pipe.unet)
    accelerator.print(
        f"Trainable parameters: {trainable:,} / {total:,} ({percentage:.4f}%)"
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    global_step = 0
    start_epoch = 0
    if config.train.resume_from:
        global_step, start_epoch = restore_checkpoint(
            pipe.unet, optimizer, config.train.resume_from
        )
        accelerator.print(f"Resumed from step {global_step} at epoch {start_epoch}")

    dataset = LatentCaptionDataset(
        config.train.latent_manifest_path, config.train.max_train_samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=accelerator.device.type == "cuda",
    )
    unet, optimizer, dataloader = accelerator.prepare(
        pipe.unet, optimizer, dataloader
    )
    optimizer.zero_grad(set_to_none=True)
    steps_per_epoch = math.ceil(
        len(dataloader) / config.train.gradient_accumulation_steps
    )
    max_steps = config.train.max_steps or config.train.epochs * steps_per_epoch
    log_path = output_root / "train_log.csv"
    if accelerator.is_main_process and not log_path.exists():
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["step", "epoch", "loss"])

    unet.train()
    last_epoch = start_epoch
    for epoch in range(start_epoch, config.train.epochs):
        last_epoch = epoch
        for latents, captions in dataloader:
            with accelerator.accumulate(unet):
                latents = latents.to(accelerator.device, dtype=torch.float32)
                conditioned = [
                    "" if torch.rand(()).item() < config.train.empty_prompt_probability else text
                    for text in captions
                ]
                with torch.no_grad():
                    prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
                        prompt=conditioned,
                        device=accelerator.device,
                        num_waveforms_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                target = diffusion_target(pipe.scheduler, latents, noise, timesteps)
                with accelerator.autocast():
                    prediction = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=generated_prompt_embeds,
                        encoder_hidden_states_1=prompt_embeds,
                        encoder_attention_mask_1=attention_mask,
                    ).sample
                    loss = F.mse_loss(prediction.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            mean_loss = accelerator.gather(loss.detach()).mean().item()
            accelerator.print(
                f"step={global_step}/{max_steps} epoch={epoch + 1} loss={mean_loss:.6f}"
            )
            if accelerator.is_main_process:
                with log_path.open("a", encoding="utf-8", newline="") as handle:
                    csv.writer(handle).writerow([global_step, epoch + 1, mean_loss])

            should_checkpoint = (
                config.train.checkpoint_every_steps > 0
                and global_step % config.train.checkpoint_every_steps == 0
            )
            if should_checkpoint:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(unet)
                    directory = save_checkpoint(
                        unwrapped,
                        optimizer,
                        output_root / "checkpoints" / f"step_{global_step:06d}",
                        global_step=global_step,
                        epoch=epoch,
                    )
                    if config.train.sample_on_checkpoint:
                        _sample_checkpoint(pipe, config, directory)
                accelerator.wait_for_everyone()

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    final_directory = output_root / "jazz_cafe_lora"
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator.unwrap_model(unet),
            optimizer,
            final_directory,
            global_step=global_step,
            epoch=last_epoch,
        )
    accelerator.wait_for_everyone()
    return final_directory
