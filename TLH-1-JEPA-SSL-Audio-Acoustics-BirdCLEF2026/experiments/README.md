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

## BirdCLEF Prediction Frame

The competition-shaped problem should be treated as window-level multi-label
classification over soundscapes:

```text
test soundscape -> fixed windows -> log-mel spectrograms
  -> shared audio encoder -> species sigmoid head
```

Each window can contain zero, one, or many species. The supervised target is
therefore a binary vector over species, not a single class index.

## Main And Auxiliary Signals

The first runnable model should keep one shared encoder and one multi-label
head. Additional signals can be attached during training without changing the
inference shape:

- **Soundscape SSL:** train the encoder on soundscape windows with masked
  latent prediction before supervised fine-tuning.
- **Species-specific audio:** train the same encoder on curated species clips
  so it learns cleaner examples of each class.
- **Prototype matching:** build one embedding prototype per species from
  labeled clips and score soundscape windows against those prototypes.
- **Eventness head:** add a binary head for biological acoustic activity versus
  background/noise.
- **Pseudo-labeling:** use high-confidence soundscape predictions as additional
  training labels for a second pass.

For the first TLH implementation, treat these as ablations:

1. Supervised baseline only.
2. Baseline plus soundscape SSL pretraining.
3. Baseline plus species prototypes.
4. Baseline plus both SSL and prototypes.

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

---

## Experiment Log

### Step 1 — Supervised Baseline (complete)

**Model:** `AudioEncoder` (ViT: patch_size=16, d=256, 4 heads, 4 layers) + `nn.Linear(256, 234)`.  
**Training:** 30 epochs, AdamW, cosine LR, BCEWithLogitsLoss, A100.  
**Result:** val AUC **0.677** (macro, species with ≥1 positive in val, threshold ≥0.5).

Key implementation notes:
- BirdCLEF labels: primary_label=1.0, secondary_labels=`secondary_weight`(0.5)
- AUC metric must use `>= 0.5` threshold (not `> 0.5`) to match Kaggle's `sum > 0` filter
- Kaggle evaluation code: `scored_columns = solution.sum(axis=0)[sums > 0]`

Checkpoint: `/workspace/tlh1/outputs/step1/checkpoints/best.pt`

---

### Step 3 — Factorize-and-Compose Audio-JEPA (in progress)

**Architecture:**
```
E_c  context encoder (trainable ViT, warm-start from step1)
E_t  target encoder (EMA copy, stop-grad, τ=0.996)
P1   SlotAttention(K=8, D=256, n_iter=3) — factorize patch tokens into K slots
P2   ComposePredictor (LayerNorm → Linear → GELU → Linear) — compose slots
Clf  SlotClassifier (Linear(D, 234), mean+max pool over K slots)
cls_head  Linear(D, 234) — CLS bypass loaded from step1 head.*
```

**Training modes:**
- Mode 1 (50% of steps): single clip → E_c → P1 slots; same clip → E_t → z_tgt; L_fact = MSE(norm_matched_slot, z_tgt)
- Mode 2 (50% of steps): x_A, x_B, x_bg → E_t → z_A, z_B, z_bg; mix → E_c → P1 → slots; Hungarian match + compose

**Loss:**
```
L = L_cls + 0.5*L_fact + 1.0*L_comp + 0.1*L_div + 0.1*L_var
```
- L_fact: cosine-distance-based MSE between **normalized** matched slots and unit-norm targets
- L_comp: MSE between **normalized** P2 output and unit-norm composition target
- L_div: off-diagonal cosine penalty in slot Gram matrix (slot diversity)
- L_var: VICReg variance on CLS token across batch (encoder anti-collapse)

**Warm start:** enc_ctx + cls_head loaded from step1 best.pt. Slot head zero-initialized (no noise at epoch 1).

**Differential LR:** enc_ctx at 3e-5 (10× lower), new heads at 3e-4.

**Current run:** PID 16554 on A100, log: `/workspace/tlh1/logs/step3_20260512_210031.log`

#### Step 3 journey and lessons

| Attempt | Issue | Fix |
|---|---|---|
| Epoch 1 AUC 0.542 | CLS gradient path abandoned — slot head with random init corrupted step1 calibration | Added `cls_head` bypass loaded from step1 weights; additive ensemble |
| AUC decaying 0.735→0.720 | Random slot head adding noise to CLS ensemble | Zero-init slot head weights/bias |
| AUC shock at epoch 6 | `freeze_enc_epochs=5` caused sudden unfreeze | Removed freeze; end-to-end with differential LR from epoch 1 |
| Stop-grad on cls experiment | AUC collapsed 0.725→0.608 in 3 epochs | JEPA reshapes encoder, cls_head calibrated for old features can't adapt fast enough. Reverted. |
| L_fact / L_comp → ~0.003 | Trivial zero-slot minimum: MSE(0, unit_vec) = 1/D ≈ 0.004 | Normalize predictions before MSE (BYOL formulation). Losses now in cosine-distance space |
| BFloat16 crash in scipy | `torch.mm` inside AMP autocast returns bfloat16; numpy doesn't support it | `.float()` on cost tensor before `.numpy()` |
| AUC metric understates valid species | `lall > 0.5` drops secondary labels (exactly 0.5) from scoring | Changed to `lall >= 0.5` to match Kaggle's `sum > 0` |

#### Next iteration ideas (after baseline)

1. **Stronger anti-collapse:** EMA target encoder collapse is the root risk. Consider gradient-variance monitoring or explicit output normalization on enc_tgt outputs.
2. **Cosine instead of MSE for L_comp:** `1 - cos_sim(norm(z_comp_hat), z_comp)` is more interpretable and lives in [0, 2].
3. **K ablation:** Try K=4, K=12. K=8 with Hungarian matching to M=3 targets leaves 5 slots unguided.
4. **Mode 3 (real soundscape):** Current Mode 2 uses synthetic audio mixing. A Mode 3 using real labeled soundscapes (pairing a window with its annotated species clips) would provide cleaner signal.
5. **Masking:** Add spectral masking (frequency bands) to the context before passing to E_c — forces P1 to predict from incomplete context, the true JEPA setup.
