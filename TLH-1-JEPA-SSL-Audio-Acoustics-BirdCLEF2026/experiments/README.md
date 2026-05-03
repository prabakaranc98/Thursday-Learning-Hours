# Experiment Plan

## Goal

Build a small JEPA-style self-supervised audio experiment and use it to reason
about BirdCLEF 2026-style species identification.

This is a teaching experiment, not a full competition pipeline.

## Pipeline

1. Load short audio clips.
2. Convert each clip to a log-mel spectrogram.
3. Split spectrograms into time-frequency patches.
4. Sample visible context patches and masked target patches.
5. Train a context encoder and predictor to match target encoder embeddings.
6. Freeze or lightly fine-tune the encoder.
7. Train a small classifier for species labels.
8. Compare against a supervised baseline trained from scratch.

## Minimal Model

- Context encoder: small CNN or patch transformer.
- Target encoder: same architecture, updated by exponential moving average.
- Predictor: small MLP or transformer block over visible context tokens.
- Loss: cosine distance or mean squared error between predicted and target
  embeddings.

## Demo Data Strategy

Preferred:

- Use a small subset of the BirdCLEF 2026 data if available locally through
  Kaggle.

Fallback:

- Use a tiny local audio subset with the same pipeline shape.
- Keep file names and labels in a local CSV under `data/`.

## Files To Add Later

- `prepare_audio.py`: build log-mel examples from raw audio.
- `pretrain_jepa.py`: run the SSL pretraining loop.
- `train_probe.py`: train and evaluate the downstream classifier.
- `models.py`: encoders, predictor, and classifier heads.
- `config.yaml`: sample rate, window length, mel bins, patch size, and masking
  policy.

## First Success Criterion

The demo should run end to end on a small local subset and produce:

- supervised baseline metric,
- JEPA-pretrained metric,
- training curves,
- short notes on observed failure modes.
