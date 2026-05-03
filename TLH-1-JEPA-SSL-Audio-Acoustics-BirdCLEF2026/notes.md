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

## Discussion Prompts

- What should count as a good acoustic representation?
- Does the model need frequency masking, time masking, or structured event
  masking?
- How do we avoid learning background and recorder artifacts instead of species
  cues?
- When would a supervised baseline beat the SSL method?
- What metrics should matter for a conservation monitoring system beyond
  leaderboard score?

## Open Items

- Confirm local compute target for the demo.
- Decide whether the first runnable version uses public BirdCLEF data directly
  or a tiny local audio subset.
- Choose the first backbone: small CNN, Audio Spectrogram Transformer, or a
  deliberately minimal patch encoder.
