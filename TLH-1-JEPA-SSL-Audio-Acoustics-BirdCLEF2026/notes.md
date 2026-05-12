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

## Open Items

- Confirm local compute target for the demo.
- Decide whether the first runnable version uses public BirdCLEF data directly
  or a tiny local audio subset.
- Choose the first backbone: small CNN, Audio Spectrogram Transformer, or a
  deliberately minimal patch encoder.
