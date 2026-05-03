# Slides Outline

## 1. Why Audio SSL?

- Long recordings, sparse labels, noisy environments.
- Passive acoustic monitoring as representation learning at scale.
- BirdCLEF 2026 as the motivating case study.

## 2. BirdCLEF 2026 Problem Frame

- Wildlife species identification from field audio.
- Pantanal wetlands and continuous recorder deployments.
- Multi-species soundscapes, limited labels, and messy background conditions.

## 3. From Audio to Learning Units

- Waveform clips.
- Windowing.
- Log-mel spectrograms.
- Time-frequency patches.

## 4. SSL Families for Audio

- Contrastive methods.
- Masked reconstruction methods.
- Predictive embedding methods.

## 5. JEPA Core Idea

- Predict latent target representations from visible context.
- Avoid raw reconstruction when low-level detail is not the objective.
- Use masking to force semantic acoustic prediction.

## 6. Audio JEPA Design

- Context encoder.
- Target encoder.
- Predictor.
- Masking policy.
- Latent prediction loss.

## 7. Mini Experiment

- Data subset.
- Baseline classifier.
- JEPA pretraining.
- Linear probe or light fine-tune.
- Comparison and caveats.

## 8. Discussion

- What would scale?
- What would fail?
- What should be measured beyond leaderboard performance?
