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

**Run:** A100, log: `/workspace/tlh1/logs/step3_20260512_210031.log` (complete, 30 epochs)  
**Best checkpoint:** `/workspace/tlh1/outputs/step3/checkpoints/best.pt` (epoch 1, AUC=0.7265)

#### Step 3 — final results (30 epochs)

| Epoch | val_auc | L_fact | L_comp | L_div | note |
|---|---|---|---|---|---|
| 1 | **0.7265** ← best | 0.0068 | 0.0037 | 0.183 | step1 warm start |
| 5 | 0.7062 | 0.0063 | 0.0014 | 0.048 | JEPA disturbs step1 features |
| 10 | 0.7115 | 0.0065 | 0.0022 | 0.040 | recovering |
| 20 | 0.7211 | 0.0065 | 0.0025 | 0.033 | slots diversifying |
| 30 | 0.7255 | 0.0065 | 0.0024 | 0.030 | stable |

Step 1 baseline: 0.677. Step 3 best: **0.7265** (+0.049 over step 1).

**Critical observation:** L_fact stabilized at 0.0065 from epoch 2 and never improved.
Random baseline in cosine-MSE space = 2/D ≈ 0.0078. So slots achieve cosine similarity
of ~0.17 with target representations — marginally better than random. Classification
(L_cls) did all the work; the JEPA factorization component is not yet contributing
meaningfully. The AUC gain is entirely from the step1 warm start and CLS bypass head.

**Submission note:** Local A100 copy has no test soundscapes. LB score requires a Kaggle
kernel that mounts competition test data. Checkpoint to upload: `best.pt` (epoch 1).

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

---

### Step 4 — Clean FC-JEPA + SIGReg (running)

**Motivation:** Step 3 had three architecture flaws identified by root-cause analysis:
1. **Mode 1 was BYOL, not JEPA** — same clip fed to both E_c and E_t, no view gap, no factorization signal
2. **Composition in spectrogram space** — `spec_mix = w₀·x_A + w₁·x_B + w₂·x_bg` mixes audio before encoding; semantic factorization requires composition in *representation space*
3. **K=8 with M=3 targets** — 5 slots received zero gradient from L_fact, causing identity collapse

**Architecture (`experiments/step4_clean_fcjepa/`):**

```
E_c  context encoder (trainable ViT)   — sees real noisy soundscape windows
E_t  target encoder (EMA, stop-grad)   — sees label-conditioned clean audio
P1   SlotAttentionV4 (K_max=6)         — per-slot learned mu (vs shared mu in step3)
P2   ComposePredictor                  — slots → predicted composition embedding
clf  SlotClassifier (max+mean pool)
cls_head  Linear(D, n_cls)             — CLS bypass, warm-started from step3
```

**Key change — view gap (true JEPA):**
```
E_c input : real soundscape window (noisy, all species mixed)
E_t input : label-conditioned clean audio per species present in that window
            → stitch random clips from train_audio/[taxon_id]/ per species
            → white noise for empty (silent) windows
```

**Key change — representation-space composition:**
```
z_comp = F.normalize(z_s1 + z_s2 + ... + z_sm, dim=-1)
```
Computed algebraically from individual species embeddings. No mixed audio ever enters E_t.

**Key change — K = M dynamic slots:**
`n_slots = n_active.max()` per batch; unused slots masked out of Hungarian matching loss.
Per-slot `slot_mu ∈ ℝ^{K×D}` (vs shared `ℝ^{1×D}`) gives each slot a distinct learned starting point.

**Key change — SIGReg (LeWorldModel arXiv:2603.19312):**
Replaces VICReg variance + slot diversity Gram loss with a single principled regularizer.
Uses Cramér-Wold theorem: a distribution is N(0,I) iff every 1D projection is N(0,1).
Moment-matching proxy (kurtosis + skewness) — O(B × n_proj), no hyperparameter tuning.
Applied to CLS tokens and all slot embeddings every step.

**Loss:**
```
L = L_cls + 1.0·L_fact + 0.5·L_comp + 0.1·L_sig

L_cls  = BCEWithLogitsLoss(cls+slot logits, y)    [labeled samples only]
L_fact = Hungarian cosine distance(slots[:n_active], {z_si})
L_comp = 1 − cosine_sim(P2(slots, acts), z_comp)  [multi-source windows only]
L_sig  = sig_reg(cls_norm) + sig_reg(slots_norm)  [all samples incl. unlabeled]
```

**Full data utilization:**

| Source | Samples | Signal |
|---|---|---|
| `train_soundscapes_labels.csv` | 638 windows (57 files) | Full JEPA: L_fact + L_comp + L_cls |
| `train_audio/` curated clips | ~30K clips (206 species) | Single-species JEPA (different crop = target) + L_cls w/ secondary labels |
| `train_soundscapes/` unlabeled | 127,896 windows (10,658 files), 8,192/epoch | SIGReg only — real Pantanal acoustics, same sites as test |

**Training:** batch=128, lr=5e-4 (linear warmup 2 epochs → cosine), A100, warm start from step3 best.pt.  
Differential LR: enc_ctx at 10× lower than new heads.

**WandB:** https://wandb.ai/karan98/fc-audio-jepa  
**Checkpoint:** `/workspace/tlh1/outputs/step4/checkpoints/best.pt`

#### Step 4 — architecture and implementation lessons

| Issue | Root cause | Fix |
|---|---|---|
| Warm start size mismatch | slot_mu shape (1,1,D)→(1,K,D) changed | Filter state dict by shape before `load_state_dict(strict=False)` |
| LR stuck at ~0 | Warmup set `initial_lr` from already-zeroed LR at step 0 | Replaced manual warmup with `SequentialLR(LinearLR, CosineAnnealingLR)` |
| `autocast` TypeError on PyTorch 2.6 | `torch.cuda.amp.autocast(device_type=...)` deprecated | Use `torch.amp.autocast(device_type, dtype=...)` |
| `GradScaler` deprecation | Same | `GradScaler("cuda", enabled=...)` |
| Overfitting (638 windows, 30 epochs) | Too few soundscape windows for full classifier training | Added 30K curated clips + 8K unlabeled soundscapes per epoch |
| NaN in L_fact at epoch 1 | bfloat16 + random slot init at warmup LR | GradScaler skips NaN steps; stable from epoch 2 |
| Host key changed after reboot | SSH reconnect failure | `ssh-keygen -R` then `StrictHostKeyChecking=no` |
