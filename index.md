# Welcome to ML_Jazz!

In this project, I will walk you through my journey in the ways of how machines generate music. The journey ended up exploring autoregressive generation with transformers, diffusion models from bare bones, and adapting a pretrained diffusion model with LoRA.

## Agenda

1. [Pretrained diffusion with LoRA finetuning (current approach)](#1-pretrained-diffusion-with-lora-finetuning-current-approach)
2. [Previous trials with diffusion](#2-previous-trials-with-diffusion)
3. [Transformer autoregressive generation](#3-transformer-autoregressive-generation)

---

## 1. Pretrained diffusion with LoRA finetuning (current approach)

The current approach adapts a pretrained text-to-audio diffusion model to the sound and phrasing of jazz while keeping the original model largely intact. LoRA makes that adaptation practical by training a small set of additional weights rather than updating the full model.

### Generated music

#### Highlights

**Saxophone highlight**

<audio src="assets/audio/lora/sample%20sax.wav" controls preload="none" style="width: 100%"></audio>

**General highlight**

<audio src="assets/audio/lora/sample.wav" controls preload="none" style="width: 100%"></audio>

#### More samples

**Piano**

<audio src="assets/audio/lora/00%20piano.wav" controls preload="none" style="width: 100%"></audio>

**Saxophone — blues**

<audio src="assets/audio/lora/00%20sax%20blues.wav" controls preload="none" style="width: 100%"></audio>

**Saxophone — swing**

<audio src="assets/audio/lora/00%20sax%20swing.wav" controls preload="none" style="width: 100%"></audio>

**Slow saxophone**

<audio src="assets/audio/lora/00%20slow%20sax.wav" controls preload="none" style="width: 100%"></audio>

### The pipeline

**Prepare data → cache latents → train → generate → evaluate**

- **Prepare data:** turn the source recordings into consistent clips, divide them into training and holdout sets, and pair them with useful text captions.
- **Cache latents:** encode the training audio once with the pretrained variational autoencoder so training does not repeat this expensive step.
- **Train:** optimize only the LoRA attention weights while keeping the pretrained AudioLDM2 model frozen.
- **Generate:** load an adapter checkpoint and synthesize music from jazz-focused text prompts.
- **Evaluate:** compare base-model and adapted generations with matched prompts and seeds, using both listening tests and quantitative measures such as Fréchet Audio Distance.

### Training details

The model was trained for **80 epochs** with a **batch size of 4**. A checkpoint was saved every **500 steps**, providing regular recovery points and listening samples for comparing how the musical character developed over time.

### Model and adapter architecture details

This project uses **AudioLDM2**, a pretrained latent diffusion model designed for text-conditioned audio generation. It provides a strong general musical prior, which means the experiment can focus on adapting style and instrumentation rather than learning audio generation from scratch. LoRA adapters are attached to the attention layers of the diffusion network while the larger pretrained pipeline remains frozen.

A **LoRA rank of 16** was selected as a practical balance: it offers enough capacity to learn the timbre and phrasing of the jazz dataset while keeping the adapter compact, reducing memory use, and limiting the tendency to overfit a comparatively focused collection of recordings.

### Trials with rank 16 and rank 32


### Learning rate tuning

The learning rate is tapered near the end of training. Early updates can make broader stylistic changes; the smaller late-stage updates help refine the adapter without abruptly disturbing musical structure that has already been learned. The final taper is also a useful safeguard against audible degradation or overfitting late in the run.

### Suggested next steps

- Compare rank 16 and rank 32 with the same prompts, seeds, checkpoints, and listening rubric so the effect of adapter capacity is isolated.
- Curate a smaller second-stage dataset around phrasing, articulation, and instrument balance, then fine-tune from the best style adapter at a lower learning rate.
- Track checkpoint quality with blind listening comparisons in addition to loss; musical quality does not always follow the training curve.
- Expand captions with tempo, instrumentation, articulation, and mood so prompt control can be evaluated more precisely.
- Report results on a fixed holdout set and a larger generated set before drawing conclusions from Fréchet Audio Distance.

---

## 2. Previous trials with diffusion

### Diffusion trial

This trial explores learning diffusion by injecting random noise into an original audio sample.

#### Original audio sample

`gen_orig_4.wav` is the original audio sample.

<audio src="assets/audio/legacy/gen_orig_4.wav" controls></audio>

#### Noisy diffusion sample

`gen_4.wav` is the sample with random noise injected while the model is in the process of learning diffusion.

<audio src="assets/audio/legacy/gen_4.wav" controls></audio>

---

## 3. Transformer autoregressive generation

Jazz is a free-flowing musical language that is rooted in swing, and improvised melodies. In this project, I aimed to have an AI generate a comprehensible and melodical piece of music, with contrasting ranges in pitch, jazz articulations, and dynamics.

### Samples from the 8 layer transformer model built from the ground up

#### Primer melody - Autumn Leaves - Chordal Melody

(A sample is provided as context and for autoregressive generation, the generated melody starts at 0:27 - it is noticeable the difference - still working on improvements!)
<audio src="assets/audio/legacy/autumn_leaves.mp3" controls></audio>

#### Blues for Alice 1 - Charlie Parker

- One can notice the attempt to mirror the embelishments that Parker puts in
<audio src="assets/audio/legacy/blues_for_alice_1.mp3" controls></audio>

Sample 2
<audio src="assets/audio/legacy/blues_for_alice_2.mp3" controls></audio>

Sample 3
<audio src="assets/audio/legacy/blues_for_alice_3.mp3" controls></audio>

#### There will never be another you - Chet Baker

- a more space out melody which the model learns to imitate
<audio src="assets/audio/legacy/never_be_another_you.mp3" controls></audio>

### Previous experimentation with a pretrained model developed by Google Magenta

### Some background:

- An Attention Recurrent Neural Network (RNN) was used to capture longer patterns within the music, allowing for better musical phrases and context. 
- The song was based on the Charlie Parker tune, Blues for Alice. Piano was selected to be the lead instrument.
- The model was trained on collected recordings on the web of pianists, as well as snippets which I personally recorded!
- Check out the GitHub for more detailed information: <a href="https://github.com/BrightLaw9/ML_Jazz" target="_blank">GitHub</a>

Take a listen to some of the generated music below! (Note: it's no where near that of a human)

### Sample 1 

<audio src="assets/audio/legacy/Blues_for_Alice_ML_v1.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 2 layer RNN with 64 processing units each
- 0:51 - 1:03 - Repetition of a common starting phase
- 1:15 - 1:16 - Blend nicely with Cm7 - F7 harmonic structure
- Notable dissonance throughout (playing E)

### Sample 2 

<audio src="assets/audio/legacy/Blues_for_Alice_ML_v2_swing.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 2 layer RNN with 128 processing units each
- Relatively more swing feel than others
- At 1:20, there is instance of overfitting
- 1:32 - interesting swing groove happening! (solid chrous of improv)
- 2:02 - fit the Abm7 -> Db7 harmony nicely
- Occasional jazz articulations/phrases appearing (one at 2:37 - 2:38)

### Sample 3 

<audio src="assets/audio/legacy/Blues_for_Alice_ML_v3_color.mp3" controls></audio>
Notes & Notable Timestamps: 
- Trained on a 3 layer RNN with 64 processing units each
- 1:20 - Interesting stepping up sequence
- The stepping up sequence transitions from mid to high range at 1:30 - 1:33
- 1:33 - 1:34 - Melodical phrase generated!
- 1:40 - 2:36 - A period of exotic back and forths with the upper and lower ranges of pitch (effects of attention)
- 2:36 - Return to natural phrasing

