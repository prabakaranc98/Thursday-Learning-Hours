# Notes

## Core Question

How do we build useful acoustic representations when labels are scarce, audio
is noisy, and the downstream task is species identification from field
recordings?

## Key Ideas

- Passive acoustic monitoring produces much more audio than humans can inspect.
- Labels are sparse, ambiguous, and often multi-label.
- SSL can use unlabeled audio to learn acoustic structure before supervised
  training.
- JEPA-style learning predicts target embeddings from context embeddings rather
  than reconstructing raw spectrogram pixels.
- For audio, target regions can be masked across time, frequency, or both.

## JEPA Sketch for Audio

1. Convert waveform clips to log-mel spectrograms.
2. Split spectrograms into time-frequency patches.
3. Sample visible context patches and masked target patches.
4. Encode visible context with a context encoder.
5. Encode target patches with a target encoder.
6. Predict target latent vectors from context latents.
7. Train with latent-space prediction loss.
8. Evaluate the encoder with a small classifier on species labels.

## BirdCLEF From First Principles

The applied task is not single-label audio classification. A test soundscape can
contain silence, background insects, weather, recorder artifacts, and multiple
species calling at overlapping or different times. The prediction unit should
therefore be a time window from the soundscape, and the target should be a
multi-label vector over species.

The cleanest mental model is:

1. Slice each long soundscape into fixed windows.
2. Convert each window into a log-mel spectrogram.
3. Use one shared acoustic encoder to produce time-frequency representations.
4. Attach a sigmoid multi-label classifier head for species presence.
5. Aggregate window-level predictions into the submission format required by
   the task.

The main encoder should learn from train soundscapes because they match the test
domain: long passive recordings, overlapping calls, environmental noise, and
recorder conditions. The species-specific train audio is still critical, but it
has a different role: it teaches the model what each species sounds like under
clearer or more curated conditions.

## Architecture Hypothesis

Use a shared audio backbone as the center of the system:

```text
waveform window
  -> log-mel spectrogram
  -> patch / CNN / transformer audio encoder
  -> multi-label species head
  -> sigmoid species probabilities
```

Then add auxiliary learning paths that make the backbone less dependent on
scarce window-level labels:

- **Self-supervised soundscape objective:** pretrain the encoder on train
  soundscape windows with masked prediction or JEPA-style latent prediction.
- **Species prototype objective:** encode labeled species clips and learn
  species prototypes; compare soundscape window embeddings against those
  prototypes.
- **Supervised clip objective:** train on species-specific audio with standard
  multi-label or single-positive binary cross-entropy, while sharing the same
  encoder.
- **Eventness objective:** predict whether a window contains a salient
  biological acoustic event before asking which species it is.
- **Domain robustness objective:** use augmentation and/or a small domain head
  so the model does not overfit to recorder, site, gain, background, or clip
  curation artifacts.
- **Pseudo-label objective:** after a first supervised model exists, assign
  high-confidence labels to unlabeled soundscape windows and train a student
  model on both real and pseudo labels.

This gives a practical training order:

1. Pretrain the encoder on unlabeled soundscape windows.
2. Train or fine-tune with species-specific labeled audio.
3. Add a multi-label head and train on any labeled soundscape windows.
4. Use species prototypes and pseudo-labels to improve recall on rare species.
5. Calibrate thresholds per species because multi-label probabilities are not
   equally reliable across common and rare classes.

The first baseline should stay simple: log-mel windows, a small audio encoder,
binary cross-entropy, and sigmoid outputs. The research layer is then to test
whether JEPA pretraining, prototypes, eventness, and pseudo-labeling improve
that baseline.

## Discussion Prompts

- What should count as a good acoustic representation?
- Does the model need frequency masking, time masking, or structured event
  masking?
- How do we avoid learning background and recorder artifacts instead of species
  cues?
- When would a supervised baseline beat the SSL method?
- What metrics should matter for a conservation monitoring system beyond
  leaderboard score?

## BirdCLEF 2026 Data Schema (confirmed)

- `taxonomy.csv`: primary_label (string), inat_taxon_id — labels are either
  numeric iNat taxon IDs (non-birds: insects, frogs, reptiles) or eBird codes (birds)
- `train.csv`: per-clip metadata — primary_label, secondary_labels (Python list str),
  filename ("label/clip.ogg"), rating, ...
- `train_audio/{primary_label}/{clip}.ogg`: curated species recordings
- `train_soundscapes/{filename}.ogg`: passive field recordings (Pantanal)
- `train_soundscapes_labels.csv`: filename, start (HH:MM:SS), end, primary_label
  (semicolon-separated species IDs per 5-second window)
- 234 species, 35,549 curated clips, 66 soundscapes, 1,478 labeled windows
- Competition data mounted at `/kaggle/input/competitions/birdclef-2026/`

## Factorize-and-Compose: Correct Training Architecture

Context encoder E_c always takes REAL soundscape windows as input.
Composition happens in LATENT SPACE — not in audio space.

Mode 3 (primary, Natural Labeled Soundscape Training):
```
soundscape window w → E_c → P1 slots → Clf → L_cls (BCE)
                              ↓
              slots → Hung. match → L_factored (vs {z_i})
              slots → P2 → z_hat → L_composed (vs z_comp)

Given labels(w) = {A, B, ...}:
  for each species i: sample random clip from train_audio/
  z_i = E_t(clip_i)           ← target encoder, EMA copy, no gradient
  z_comp = norm(Σ z_i)        ← latent composition (NOT audio mixing)
```

Mode 2 (synthetic, augmentation only):
```
x_mix = mix(x_A, x_B, x_bg)  ← AUDIO mixing, for controlled training
x_mix → E_c → P1 → slots → losses (same as Mode 3)
z_comp = norm(z_A + z_B + z_bg) from E_t of individual clips
```

Step 1 baseline uses Mode 3 without the JEPA part: just BCE on both datasets.
Pairing soundscape windows with their species clips is needed for Step 3+.

## K Factors: Design Decision

K (number of slots) is a fixed hyperparameter — NOT variational, NOT equal to N or 234.

K should be slightly larger than the typical number of species per soundscape window.
BirdCLEF 2026 soundscape windows typically have 3–8 labeled species → K=8 or K=12.

Why K ≠ 234:
- Hungarian matching over 234×234 is expensive
- Most slots empty per window → sparse gradients
- Would become a species-indexed prototype bank, not a learned factorizer

Why K ≠ N (variable):
- Hungarian matching handles K > N: extra slots learn background/noise/silence
- Simpler implementation — no variable-length batching

The classifier reads all K slots and maps to 234 species via a linear head.
K=8 "capacity" is enough for a 234-class output head.

Ablation plan: K = 4, 8, 12, 16 after initial runs at K=8.

## Patch Sparsity: Handling Windows Without Clear Acoustic Events

This is the core challenge acoustic JEPA must solve. A 5-second soundscape window
sliced into 16×16 patches (on a 128×501 log-mel) gives ~192 patches. In a typical
window, only 5–15 patches will contain a bird call; the rest are background noise,
silence, or environmental texture.

Two distinct sparsity problems:

**1. Window-level sparsity** — many 5-second windows contain no labeled species at all.
   Even windows with a label may have the call only in 1–2 seconds of it.

**2. Patch-level sparsity** — within an active window, the signal patches are a
   small minority. A naive masked autoencoder trained on random mask patterns
   spends most of its loss on predicting silence or noise.

### Why JEPA is naturally more robust than pixel-reconstruction

Standard MAE predicts raw spectrogram values for masked patches. Predicting
background texture is just as expensive in the loss as predicting a call, so the
model wastes capacity learning noise statistics. JEPA predicts in **latent space** —
the target encoder naturally collapses similar (silent/noise) patches into a compact
region, so correctly predicting silence gets low gradient. The representation space
does the work of denoising.

### Strategies for the FC-Audio-JEPA pipeline

**Step 2 (masked JEPA pretraining) — energy-guided masking:**
Instead of uniform random masking, sample patches to include in the context with
probability proportional to their energy (softmax over log-mel amplitude per patch).
Effect: high-energy patches (likely calls) are preferentially in the context; the
predictor must predict the acoustic content, not silence. Implementation:

```python
# energy per patch: mean of log-mel values in the patch region
energy = patch_logmel.mean(dim=(-1,-2))   # (N_patches,)
weights = torch.softmax(energy / temp, dim=0)
context_idx = torch.multinomial(weights, n_context, replacement=False)
```

**Step 3 (FC-JEPA fine-tuning) — window activity gating:**
For the factorisation loss L_fact, weight each window by its acoustic activity.
Windows where all patches are below an energy threshold contribute zero gradient
for L_fact and L_comp; they only contribute to L_cls (which correctly gets 0-label
BCE for absent species). In practice, keep a simple gate:

```python
window_activity = logmel.max().item()   # peak energy in window
fact_weight = min(1.0, max(0.0, (window_activity - silence_db) / 20.0))
L = L_cls + fact_weight * (lambda1 * L_fact + lambda2 * L_comp)
```

**Slot attention naturally handles patch sparsity:**
P1's cross-attention queries (slots) compete for patches via softmax attention.
In a sparse window, background patches group into one or two background slots.
The K > N unmatched slots (where N = number of labeled species) have no factorisation
target — they learn from L_comp and L_cls only. Empirically the background slot
should emerge without any special loss term; if it doesn't, a diversity penalty on
attention distributions can help.

**Patch-window consistency at inference:**
At test time, run the model on overlapping 5-second windows (2.5 s stride) and
average predictions before thresholding. This recovers calls that happen near the
boundary of a non-overlapping window. Cost: 2× forward passes per file.

### What NOT to do

- Do not filter silent windows entirely from training: the model needs negative
  examples (all-zero labels) to calibrate its detection threshold.
- Do not mix silent patches into the positive-slots training: Hungarian matching
  already handles K > N by leaving slots unmatched.
- Do not use raw spectrogram energy as a hard gate: some species (e.g. very high
  frequency insects) have low average energy but sharp tonal peaks; use max-energy,
  not mean.

### Summary: no special windowing needed for Step 1

The baseline (Step 1) is supervised BCE; L_fact and L_comp don't exist yet.
The soft label (0.75 vs 1.0) already encodes "maybe only part of the window has
this species." The energy-guided masking is only introduced in Step 2 pretraining.

## Open Items

- Confirm local compute target for the demo.
- Decide whether the first runnable version uses public BirdCLEF data directly
  or a tiny local audio subset.
- Choose the first backbone: small CNN, Audio Spectrogram Transformer, or a
  deliberately minimal patch encoder.
