# AudioLDM2 jazz-cafe LoRA

> [!NOTE]
> **Check out the website for more information (including generated music): [https://brightlaw9.github.io/ML_Jazz_v2/](https://brightlaw9.github.io/ML_Jazz_v2/)**

This repository implements the workflow in `audioldm2_jazz_lora_finetuning.md` as a reproducible pipeline. It streams the large recordings into clips, creates a block-based train/holdout split, assigns instrumentation-aware captions, caches VAE latents, trains only LoRA attention weights, stores resumable checkpoints, and generates matched base/LoRA evaluation audio.

## Environment

Use a normal CPython 3.10–3.12 installation. The existing `.venv` in this workspace was created from LibreOffice's embedded Python and contains no Python executable, so it must be recreated before these commands will work.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of matching `torch` and `torchaudio` first, using the command from PyTorch's installer page, then install `requirements.txt`.

## CPU direction smoke test

The first cache command downloads only the VAE. Training/generation downloads the full `cvssp/audioldm2-music` pipeline and may use roughly 20 GB of system RAM. CPU inference is slow; the smoke config deliberately runs only two optimizer steps and does not generate checkpoint audio.

```powershell
python -m scripts.prepare_data --config configs\cpu_smoke.json --max-clips 12
python -m scripts.cache_latents --config configs\cpu_smoke.json --device cpu --max-samples 2
python -m scripts.reconstruct_latent --config configs\cpu_smoke.json --device cpu --output output\smoke\reconstruction.wav
accelerate launch --cpu --module scripts.train_lora --config configs\cpu_smoke.json
python -m scripts.generate --config configs\cpu_smoke.json --device cpu --adapter output\smoke\jazz_cafe_lora --steps 10 --prompt "A beautiful jazz piano melody." --output output\smoke\sample.wav
```

Listen to `output/smoke/reconstruction.wav` before training. It should clearly contain the source music; severe noise, silence, or time/frequency distortion means preprocessing must be fixed before proceeding. Two training steps only verify data flow, finite loss, gradients, checkpoint saving, and adapter loading—they cannot demonstrate adaptation.

## Full run

```powershell
python -m scripts.prepare_data --config configs\train.json
python -m scripts.cache_latents --config configs\train.json --device cuda
accelerate launch --module scripts.train_lora --config configs\train.json
```

Artifacts are written to:

- `data/clips/train` and `data/clips/holdout`: 16 kHz mono clips.
- `data/latents`: cached training latents only; holdout audio is never cached for training.
- `output/checkpoints/step_*`: safetensors LoRA weights, raw PEFT/resume state, metadata, and optional listening sample.
- `output/jazz_cafe_lora`: final adapter.
- `output/train_log.csv`: per-update loss history.

To resume an interrupted run on the same dataset, set `train.resume_from` to a
checkpoint directory. Its `max_steps` is an absolute total, so it must be larger
than the checkpoint's saved `global_step`. Keep the model and optimizer settings
unchanged.

For a second phase on a smaller, curated phrasing dataset, initialize from the
jazz-style adapter but begin a new training run instead of resuming its progress:

```json
"train": {
  "output_dir": "output/phrasing",
  "init_adapter_from": "output/jazz_cafe_lora",
  "resume_from": null,
  "max_steps": 300,
  "learning_rate": 0.000003,
  "checkpoint_every_steps": 50
}
```

Retain the other existing `train` fields. `init_adapter_from` loads only the LoRA
weights; optimizer state, epoch, and `global_step` restart at zero. The LoRA rank,
alpha, and target modules must match the source adapter. Use a separate
`output_dir` so the first-stage checkpoints and logs are not overwritten.

## Judging the first run

After verifying a finite smoke loss, run enough GPU steps to create the 500 and 1000-step samples. A useful early run should shift timbre toward the caption while retaining clean musical audio. If loss is finite but samples remain unchanged, try `learning_rate: 3e-5`; if samples become repetitive or degraded, stop at an earlier checkpoint or use rank 8. Do not infer quality from loss alone.

Generate a matched control set with identical prompts and random seeds, then optionally compute FAD:

```powershell
pip install -r requirements-eval.txt
python -m scripts.evaluate --config configs\train.json --adapter output\jazz_cafe_lora --device cuda --count 30 --steps 100
python -m scripts.evaluate --config configs\train.json --adapter output\jazz_cafe_lora --device cuda --count 100 --steps 200 --fad
```

Thirty samples are only a listening/pipeline check. Prefer 100 or more per condition before interpreting FAD.

## SLURM

The included `slurm_train.sh` prepares data, caches latents, and trains on one
RTX 6000 using `configs/train.json`. Create the log directory before
submission because Slurm opens its output files before the script starts:

```bash
mkdir -p logs
sbatch slurm_train.sh
```

Run the matched base/LoRA generation and 30-sample FAD evaluation as a
separate GPU job after training finishes:

```bash
mkdir -p logs
sbatch slurm_fad.sh
```
