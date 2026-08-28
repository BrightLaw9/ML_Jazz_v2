# Fine-Tuning AudioLDM2 with LoRA on Jazz Cafe Audio

**Goal:** Adapt the pretrained AudioLDM2 text-to-audio diffusion model to generate jazz cafe
music (piano, bass, drums) using LoRA on the UNet cross-attention layers.
**Data:** ~10 hours of jazz cafe audio.
**Compute:** single GPU, mixed precision.

---

## 1. Libraries

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers>=0.27.0 transformers accelerate peft
pip install datasets soundfile librosa
pip install bitsandbytes   # optional: 8-bit AdamW to save more memory
```

| Library | Role |
|---|---|
| `torch` / `torchaudio` | tensor ops, audio I/O, resampling |
| `diffusers` | `AudioLDM2Pipeline`, UNet, VAE, scheduler, LoRA adapter plumbing |
| `transformers` | CLAP + FlanT5 text encoders, GPT2 LM, SpeechT5 vocoder (all bundled in the pipeline) |
| `peft` | `LoraConfig`, low-rank adapter injection |
| `accelerate` | mixed-precision + device management training loop |
| `datasets` | audio dataset loading/batching |
| `librosa` / `soundfile` | chunking, resampling, waveform I/O |

---

## 2. Key classes & methods

```python
from diffusers import AudioLDM2Pipeline, DDIMScheduler
from peft import LoraConfig
from accelerate import Accelerator
import torch, torchaudio
```

- `AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music")` — loads UNet, VAE,
  CLAP text encoder, FlanT5 text encoder, GPT2 LM, projection model, vocoder as one object.
- `pipe.unet` — the `UNet2DConditionModel` you'll attach LoRA to.
- `pipe.vae.encode(...).latent_dist.sample()` — encodes a mel-spectrogram to the diffusion latent space (training target).
- `LoraConfig(r, lora_alpha, target_modules, init_lora_weights)` — defines the adapter.
- `unet.add_adapter(lora_config)` — injects LoRA into the specified modules (PEFT-backed, native in `diffusers`).
- `pipe.scheduler.add_noise(latents, noise, timesteps)` — forward diffusion step for training.
- `unet(noisy_latents, timesteps, encoder_hidden_states=..., encoder_hidden_states_1=...).sample` — noise prediction (AudioLDM2 UNet takes **two** cross-attention conditioning inputs — CLAP/FlanT5 projection and GPT2 LM output).
- `pipe.save_lora_weights(...)` / `pipe.unet.load_attn_procs(...)` — save/reload the trained adapter only (a few MB).

---

## 3. Implementation procedure

### Step 1 — Curate data

```python
# Directory of raw jazz cafe recordings (wav/flac), ~10 hours total
import glob, soundfile as sf

raw_files = glob.glob("data/training_samples/*.wav")
print(f"{len(raw_files)} source files")
```

**Note that the instrumentation for the songs are as follows:**
'Positive Mood Jazz' is a jazz piano trio song.
'A Quiet Night Above Manhattan' is saxophone and jazz rhythm section backing

**Clip length is worth treating as an actual experiment, not a fixed default.** 10s
matches AudioLDM2's default `audio_length_in_s` and is a solid baseline, but it caps
what the model can learn to *musical texture* ("this sounds like jazz") rather than
*musical structure* ("this piano phrase resolves into this chord while the bass holds
this groove"). Longer clips give the UNet's attention more temporal context to learn
those relationships — but cost more VRAM per sample (longer time axis in the mel/latent)
and give you fewer clips from the same 10 hours of source audio, so it's a real tradeoff,
not a free upgrade.

- **10s** — baseline. Cheapest, most clips (~3,600), safest starting point.
- **20s** — a reasonable middle ground if 10s clips sound directionless/note-salad in Step 6 listening tests.
- **30s** — most context, but roughly 3x the per-sample compute/memory of 10s, and only ~1,200 clips from the same source data. Don't jump here first — try it only if 10s clearly isn't capturing phrase-level structure.

Even at longer clip lengths, treat **coherence over 20–60 seconds** (a musical idea
actually developing, versus sounding like disconnected snippets stitched together) as
possible but not guaranteed — LoRA fine-tuning nudges a pretrained model's style and
local texture; it doesn't rearchitect its long-range temporal modeling. If longer clips
don't produce more coherent structure, that's a sign you're bumping into what LoRA on
this backbone can do, not a training bug to chase.

```python
import librosa, os

os.makedirs("data/clips", exist_ok=True)
CLIP_SEC = 10  # try 20 or 30 as a follow-up experiment, per above
SR = 16000  # AudioLDM2 native sample rate

for f in raw_files:
    y, _ = librosa.load(f, sr=SR, mono=True)
    n_clips = len(y) // (CLIP_SEC * SR)
    for i in range(n_clips):
        clip = y[i * CLIP_SEC * SR:(i + 1) * CLIP_SEC * SR]
        sf.write(f"data/clips/{os.path.basename(f)}_{i}.wav", clip, SR)
```

~10 hours at 10s/clip yields ~3,600 training clips — plenty for LoRA-scale adaptation.
(At 20s: ~1,800 clips. At 30s: ~1,200 clips.)

**Split off a held-out set now**, before any latent caching or training touches these
files — this becomes the reference distribution for Step 6's FAD/KAD comparison, and
needs to be clips the model never saw:

```python
import random

all_clips = glob.glob("data/clips/*.wav")
random.seed(42)
random.shuffle(all_clips)

n_holdout = int(0.1 * len(all_clips))
holdout_clips = all_clips[:n_holdout]
train_clips = all_clips[n_holdout:]

os.makedirs("data/holdout", exist_ok=True)
for c in holdout_clips:
    os.rename(c, c.replace("data/clips", "data/holdout"))

print(f"{len(train_clips)} train clips, {len(holdout_clips)} held out")
```

Everything from Step 2 onward (latent caching, captioning, training) should only ever
touch `data/clips/` — `data/holdout/` stays untouched until Step 6.

### Step 2 — Preprocess (waveform → mel-spectrogram → VAE latent)

AudioLDM2 trains in the VAE's latent space, not raw waveform or raw mel directly.

Load the pipeline once, up front — everything downstream (VAE, UNet, text encoders)
comes from this object:

```python
from diffusers import AudioLDM2Pipeline
import torch

pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float32)
```

Note: load in **fp32 here**, not fp16 — see the precision note in Step 5 for why.

**Important:** `pipe.feature_extractor` (a `ClapFeatureExtractor`) is *not* the right
tool here — per the diffusers docs it exists to pre-process **generated** audio back
into mel for CLAP-based automatic scoring, and it defaults to a 48kHz sampling rate
with CLAP's own "fusion" truncation strategy. Neither matches what the VAE was
actually trained to encode (16kHz audio, AudioLDM's own mel config). Compute the mel
directly instead, mirroring the original AudioLDM preprocessing:

```python
import torchaudio

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=160,
    n_mels=64,
    f_min=0,
    f_max=8000,
    power=1.0,
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)

def to_latent(waveform, vae, device):
    mel = mel_transform(waveform)           # (1, n_mels, time)
    mel = amplitude_to_db(mel)
    mel = mel.unsqueeze(0).to(device, dtype=vae.dtype)  # add channel dim -> (1, 1, n_mels, time)
    latent = vae.encode(mel).latent_dist.sample() * vae.config.scaling_factor
    return latent
```

`n_fft=1024, hop_length=160, n_mels=64, f_max=8000` matches the config the original
AudioLDM/AudioLDM2 codebase trains its VAE against at 16kHz. Verify the exact values
against `cvssp/audioldm2-music`'s config before a full run — if the checkpoint's
`vae.config` or model card specifies different mel dimensions, match those exactly,
since even small mismatches here will produce latents the pretrained VAE doesn't
decode cleanly.

Precompute and cache latents for all clips once (avoids re-running the VAE every epoch):

```python
import torch

os.makedirs("data/latents", exist_ok=True)
vae = pipe.vae.to("cuda").eval()

for clip_path in glob.glob("data/clips/*.wav"):
    y, _ = librosa.load(clip_path, sr=SR, mono=True)
    wav = torch.tensor(y).unsqueeze(0)
    with torch.no_grad():
        latent = to_latent(wav, vae, "cuda")
    torch.save(latent.cpu(), clip_path.replace("clips", "latents").replace(".wav", ".pt"))
```

### Step 3 — Freeze the backbone, attach LoRA

```python
from peft import LoraConfig

unet = pipe.unet
unet.requires_grad_(False)  # freeze everything first

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # cross-attn projections
)
unet.add_adapter(lora_config)

# Confirm only LoRA params are trainable
trainable = [p for p in unet.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
n_total = sum(p.numel() for p in unet.parameters())
print(f"Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")
```

**Correction to the earlier explanation:** `target_modules=["to_k", "to_q", "to_v",
"to_out.0"]` matches by module name, and AudioLDM2's self-attention blocks (`attn1`)
use the *same* projection names as its cross-attention blocks (`attn2`/`attn3`) — so
this config attaches LoRA to **both** self- and cross-attention, not cross-attention
only as I described earlier. This is actually standard practice (it's what most SD/SDXL
LoRA fine-tunes do) and works fine in practice, but if you want strictly cross-attention-only
adaptation, target by full module path instead:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=r".*attn2\.(to_k|to_q|to_v|to_out\.0)$|.*attn3\.(to_k|to_q|to_v|to_out\.0)$",
)
```

Start with the simple all-attention version above; only switch to the regex if you
observe timbre/synthesis quality degrading (a sign self-attention is being pulled
too far from its pretrained state).

Also freeze the VAE, CLAP/FlanT5 text encoders, GPT2 LM, and vocoder — they stay
fully fixed for this fine-tune:

```python
for m in [pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.language_model, pipe.vocoder]:
    m.requires_grad_(False)
    m.eval()
```

### Step 4 — Caption the data

A single fixed caption is the weakest part of a first-draft plan — the model has
nothing to differentiate between clips on, and risks binding the whole jazz-cafe sound
to one literal string rather than learning the underlying musical attributes. Use a
pool of captions that actually vary in mood, tempo, and texture, and assign each clip
one at random (or match by ear if you want to hand-curate):

```python
CAPTIONS = [
    "intimate jazz cafe, mellow piano trio, upright bass, brushed drums",
    "late night jazz cafe, soft piano chords, walking upright bass, brushes",
    "relaxed jazz piano trio, slow swing, warm room ambience",
    "smoky jazz cafe, sparse piano improvisation, upright bass, soft drums",
    "mellow instrumental jazz, medium tempo swing, piano trio",
    "warm intimate jazz, lyrical piano melody, walking bass, brushed percussion",
]

import random
captions = {clip: random.choice(CAPTIONS) for clip in glob.glob("data/clips/*.wav")}
```

If you can tell your clips apart by ear (uptempo vs. ballad, sparse vs. dense
comping), match captions to actual clip content rather than assigning randomly —
random assignment still teaches "jazz cafe" as a category, but matched captions teach
the model to distinguish the attributes within it (tempo, density, mood), which is
more useful if you want prompt-level control at inference.

### Step 5 — Train

```python
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader

class LatentCaptionDataset(Dataset):
    def __init__(self, latent_dir, captions):
        self.paths = glob.glob(f"{latent_dir}/*.pt")
        self.captions = captions

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        latent = torch.load(self.paths[idx])
        clip_name = os.path.basename(self.paths[idx]).replace(".pt", ".wav")
        caption = self.captions.get(f"data/clips/{clip_name}", CAPTIONS[0])
        return latent.squeeze(0), caption

dataset = LatentCaptionDataset("data/latents", captions)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

accelerator = Accelerator(mixed_precision="bf16")
optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=1e-2)

unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
unet.train()

noise_scheduler = pipe.scheduler
text_encoder, text_encoder_2, language_model = pipe.text_encoder, pipe.text_encoder_2, pipe.language_model
tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
projection_model = pipe.projection_model

NUM_EPOCHS = 10
EMPTY_PROMPT_PROB = 0.1  # fraction of steps trained with a blank caption

# Save + generate a listening sample at these step counts, so you can hear how the
# adapter is progressing rather than waiting for training to finish before finding out
# it went sideways. Adjust the list to your actual steps-per-epoch (3600 clips /
# batch_size 4 = ~900 steps/epoch, so this range covers roughly epochs 1 through 10).
CHECKPOINT_STEPS = {500, 1000, 2000, 4000, 6000, 9000}

def save_checkpoint_and_sample(unet, accelerator, pipe, step):
    from peft.utils import get_peft_model_state_dict

    ckpt_dir = f"output/checkpoints/step_{step}"
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
    pipe.save_lora_weights(ckpt_dir, unet_lora_layers=lora_state_dict)

    # Quick inference pass with the adapter at this point in training. Reuses the
    # in-memory pipe/unet rather than reloading from disk.
    unet.eval()
    with torch.no_grad():
        audio = pipe(
            prompt="intimate jazz cafe, mellow piano trio, upright bass, brushed drums",
            negative_prompt="low quality, noisy",
            num_inference_steps=100,
            audio_length_in_s=10.0,
        ).audios[0]
    unet.train()

    import scipy, os
    os.makedirs(ckpt_dir, exist_ok=True)
    scipy.io.wavfile.write(f"{ckpt_dir}/sample.wav", rate=16000, data=audio)
    print(f"  -> saved checkpoint + sample at step {step}")

global_step = 0
for epoch in range(NUM_EPOCHS):
    for latents, captions_batch in dataloader:
        latents = latents.to(accelerator.device)

        # Randomly blank out captions so the model also learns the unconditional
        # pathway -- required for classifier-free guidance (negative_prompt,
        # guidance_scale) to do anything meaningful at inference time.
        captions_batch = [
            "" if torch.rand(1).item() < EMPTY_PROMPT_PROB else c for c in captions_batch
        ]

        # Encode conditioning text (CLAP + FlanT5 -> projection -> GPT2 LM), all frozen
        with torch.no_grad():
            cond = pipe.encode_prompt(
                prompt=list(captions_batch),
                device=accelerator.device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds, attention_mask, generated_prompt_embeds = cond

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        if global_step in CHECKPOINT_STEPS:
            save_checkpoint_and_sample(unet, accelerator, pipe, global_step)

    print(f"epoch {epoch}: loss {loss.item():.4f}")

# Final adapter, in addition to the intermediate checkpoints above
from peft.utils import get_peft_model_state_dict

unwrapped_unet = accelerator.unwrap_model(unet)
lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
pipe.save_lora_weights("output/jazz_cafe_lora", unet_lora_layers=lora_state_dict)
```

Listen through the checkpoints in step order (500 → 1000 → 2000 → 4000 → 6000 → 9000).
What you're listening for at each stage:

- **Early (500–1000):** style should already be shifting toward jazz-cafe timbre —
  if it still sounds like the base model's generic output, the LoRA isn't learning
  fast enough (check LR, or that `trainable` params are actually receiving gradients).
- **Mid (2000–4000):** this is usually where you can judge whether the model is
  learning genuine jazz phrasing or just mimicking surface texture (reverb, room tone)
  without real harmonic/rhythmic structure.
- **Late (6000–9000):** watch for overfitting — samples that get *more* repetitive or
  loop-y rather than more coherent are a sign to stop training or drop the LoRA rank,
  not a sign to keep going.

### Step 6 — Evaluate

```python
# Reload base pipeline + adapter for inference
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("output/jazz_cafe_lora")

audio = pipe(
    prompt="jazz cafe music, piano trio, walking bass, brushed drums",
    negative_prompt="low quality, noisy",
    num_inference_steps=200,
    audio_length_in_s=10.0,
).audios[0]

import scipy
scipy.io.wavfile.write("output/sample.wav", rate=16000, data=audio)
```

For quantitative evaluation:

```bash
pip install laion-clap frechet_audio_distance
```

- **CLAP score**: per-sample, no reference audio needed — cosine similarity between
  a *generated* clip's CLAP embedding and its *own text prompt's* CLAP embedding.
  Measures whether the audio matches what you asked for.
- **FAD / KAD**: distributional, needs a reference set — the Fréchet/Kernel distance
  between the embedding statistics of a batch of *generated* clips and the embedding
  statistics of `data/holdout/` (the real clips split off in Step 1, never trained on).
  Measures whether the fine-tuned model's overall sound matches real jazz cafe audio,
  independent of any one prompt.

Concretely, generate a batch spanning your caption pool (not just one prompt) against
the same held-out set, and — as a control — run the same prompts through the
**un-adapted base pipeline** so you have a number to show the LoRA actually improved
things, not just a single absolute score with nothing to compare it against:

```python
from frechet_audio_distance import FrechetAudioDistance

fad = FrechetAudioDistance(model_name="clap", sample_rate=16000)

# 1. Generate N samples from the fine-tuned (LoRA) pipeline across varied prompts
pipe_lora = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16).to("cuda")
pipe_lora.load_lora_weights("output/jazz_cafe_lora")

os.makedirs("output/eval/lora_samples", exist_ok=True)
for i in range(30):
    prompt = random.choice(CAPTIONS)
    audio = pipe_lora(prompt, negative_prompt="low quality, noisy",
                       num_inference_steps=200, audio_length_in_s=10.0).audios[0]
    scipy.io.wavfile.write(f"output/eval/lora_samples/{i}.wav", rate=16000, data=audio)

# 2. Control: same prompts, base model, no adapter
pipe_base = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", torch_dtype=torch.float16).to("cuda")

os.makedirs("output/eval/base_samples", exist_ok=True)
for i in range(30):
    prompt = random.choice(CAPTIONS)
    audio = pipe_base(prompt, negative_prompt="low quality, noisy",
                       num_inference_steps=200, audio_length_in_s=10.0).audios[0]
    scipy.io.wavfile.write(f"output/eval/base_samples/{i}.wav", rate=16000, data=audio)

# 3. FAD against the same held-out reference for both -- the LoRA number should be
# meaningfully lower than the base-model number if fine-tuning actually helped.
fad_lora = fad.score("data/holdout", "output/eval/lora_samples")
fad_base = fad.score("data/holdout", "output/eval/base_samples")
print(f"FAD (LoRA):  {fad_lora:.3f}")
print(f"FAD (base):  {fad_base:.3f}")
```

If `fad_lora` isn't clearly lower than `fad_base`, that's a real signal — either the
adapter under-trained (check the checkpoint listening tests from Step 5) or the base
model was already close enough to jazz-cafe style that this LoRA isn't adding much.

**Caveat on sample size:** FAD estimates a full covariance matrix over the embedding
dimension (~512-d for CLAP) from your generated batch — 30 samples is not enough for
that estimate to be stable, and FAD is known to be noisy/biased at small N. Treat 30 as
illustrative; use 100+ generated samples per condition (and as much of the holdout as
you can spare) before trusting the numbers. If you're stuck data-constrained, compute
KAD alongside FAD — it's kernel-based (MMD) rather than a Gaussian covariance fit, so
it degrades more gracefully at low sample counts:

```python
kad_lora = fad.score("data/holdout", "output/eval/lora_samples", dtype="float32")  # KAD variant per frechet_audio_distance docs
```

Treat a clearly-lower LoRA score as a good sign, but don't treat a marginal or
ambiguous result as proof the fine-tune failed — cross-check against the Step 5
listening checkpoints before concluding anything from the number alone.

---

## Notes

- Start with `r=16`; drop to `r=8` if you see overfitting (loss keeps dropping but
  generated samples sound repetitive/degenerate given only 10 hours of data).
- If GPU memory is tight even with LoRA, add `pipe.enable_gradient_checkpointing()`
  before training, or drop `batch_size` to 2 and increase gradient accumulation.
- The `cvssp/audioldm2-music` checkpoint (rather than the general `cvssp/audioldm2`)
  is pretrained specifically on music, which is the better starting point here.
- **Precision:** load the pipeline in fp32 (Step 2) and let `Accelerator(mixed_precision="bf16")`
  handle autocasting during training, rather than loading the base model directly in
  fp16. Loading fp16 weights *and* wrapping with Accelerate's mixed-precision handling
  stacks two separate precision-management systems and risks dtype mismatches between
  the frozen fp16 backbone and fp32-initialized LoRA matrices. bf16 (used here) also
  avoids the gradient-underflow issues fp16 causes for the small LoRA weight updates,
  and doesn't need loss scaling. Switch back to fp16 only at inference time (Step 6),
  where there's no gradient to underflow.
- **Held-out set:** before Step 5, split off ~10% of `data/clips/*.wav` (and their
  cached latents) into a directory the training `DataLoader` never reads from — Step 6's
  FAD/KAD comparison needs real clips the model never saw, and there's no split in
  Step 1's chunking code as written, so add one explicitly, e.g. shuffle the file list
  once and slice it before training starts.
- **Conditioning dropout:** Step 5 now randomly blanks ~10% of captions during training —
  without this, `negative_prompt` and `guidance_scale` at inference (Step 6) have little
  effect, since the model would never have learned what "no conditioning" looks like.
