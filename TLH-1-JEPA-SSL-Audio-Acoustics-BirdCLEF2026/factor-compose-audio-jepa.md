# Factorize-and-Compose Audio-JEPA — Design Specification

**Status:** architecture locked, Step 1 baseline running, Step 3 implementation in progress  
**Updated:** 2026-05-12

---

## Problem

A passive recorder in the Pantanal captures everything: frog calls, insects, wind,
rain, and silence — all mixed together. Every 5-second window can contain zero, one,
or several species calling simultaneously. The task is a binary prediction over 234
species: *which are calling right now?*

Two datasets exist:

| Dataset | What it contains | Role in our system |
|---|---|---|
| `train_soundscapes/` | Long passive field recordings | Context path — what the model sees at inference |
| `train_audio/` | Curated single-species clips | Target path — what each species sounds like |
| `train_soundscapes_labels.csv` | Per-5s-window species labels | Supervision bridge |

The fundamental mismatch: labels are at the *window* level but species may call for
only 0.3 s of a 5 s window. Curated clips don't match soundscape acoustics.
Most species are rare.

---

## Core Idea

A soundscape window containing species A and B is a mixture of two acoustic sources.
In a good representation space the mixture latent should be predictable from A and B
individually — not from raw audio mixing, but from **composition in latent space**:

```
normalize(z_A + z_B)  ≈  latent of "A and B together"
```

We use this as a training signal without any contrastive pairs or negatives.

---

## Architecture

```
                  CONTEXT PATH (gradients flow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
soundscape window w
        │
        ▼
   S(w) = LogMel(w)              (1, F, T)  F=64, T≈500
        │
        ▼
   h = f_θ(S(w))                (N_patches, D)   N=128, D=128
        │                       one token per 16×16 patch
        ▼
   P1: Slot Attention            K=8 slots compete for N patches
        │
        ├──→  s₁…s₈             (K, D)  factored acoustic slots
        │
        ├──→  P2(slots)  →  ẑ_comp   (D,)  predicted mixture latent
        │
        └──→  Clf(slots) →  logits   (234,) species presence


                  TARGET PATH (stop-gradient always)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
given labels Y(w) = {y₁, y₂, …, yN}:
  for each yᵢ ∈ Y(w):
    xᵢ ~ random clip from train_audio/yᵢ/
    zᵢ = normalize( sg( f_θ̄( S(xᵢ) ) ) )    (D,)  unit norm

  z_comp = normalize( Σᵢ zᵢ )               (D,)  target composition
```

### Notation

| Symbol | Meaning |
|---|---|
| `f_θ` | context encoder — trainable, gradients flow |
| `f_θ̄` | target encoder — EMA copy of `f_θ`, **stop-gradient always** |
| `h` | patch token sequence: `f_θ(S(w)) → (N_patches, D)` |
| `P1` | slot attention predictor: `h → (K, D)` |
| `sₖ` | slot vector for slot k, shape `(D,)` |
| `P2` | composition predictor: `{sₖ} → ẑ_comp ∈ ℝᴰ` |
| `Clf` | classifier: `{sₖ} → logits ∈ ℝ²³⁴` (sigmoid multi-label) |
| `zᵢ` | target species latent — unit norm, stop-gradient |
| `z_comp` | target composition — unit norm, stop-gradient |
| `π*` | Hungarian assignment: which slot matches which species |

---

## Three Losses

All three are regression — no negatives, no contrastive pairs.

### L_cls — Multi-Label Classification

```
L_cls = BCE( logits, multihot_Y(w) )
```

Standard binary cross-entropy. Provides strong gradient from day one. Soft labels:
clip primary = 1.0, secondary = 0.5, soundscape label = 0.75.

### L_fact — Factorization (Hungarian-Matched)

```
π* = argmin_π  Σᵢ ‖normalize(s_{π(i)}) − zᵢ‖²   (Hungarian assignment)

L_fact = (1/N) Σᵢ MSE( s_{π*(i)},  zᵢ )
```

- Normalize slots **only for cost computation** — raw un-normalized slots go into MSE
- Unmatched slots (K > N) receive no L_fact gradient; they learn background through L_comp and L_cls
- K=8 > typical N=3–5 species per window; extra slots absorb background/silence naturally

### L_comp — Composition Consistency

```
L_comp = MSE( ẑ_comp,  z_comp )
```

P2 must predict the normalized sum of all species latents. `z_comp` is unit-normalized;
`ẑ_comp` is **not** — the norm mismatch (asymmetry) prevents representational collapse
without any explicit constraint. Same BYOL mechanism.

### Combined Loss

```
L = L_cls  +  λ₁·L_fact  +  λ₂·L_comp        λ₁ = λ₂ = 0.5
```

---

## Key Design Decisions (Settled)

### 1. No species bank. Fresh clips every step.

Each training step samples a random clip from `train_audio/` for each species in Y(w),
runs it through `f_θ̄` fresh, and builds `zᵢ` on the fly. No prototype bank, no
pre-computed cache.

**Why:** A bank decouples the target encoder from the context encoder. Fresh clips
keep the target space moving with EMA — targets get better as the target encoder improves.
The many-to-one problem (many clips per species) collapses cleanly: z_comp is the
normalized sum of whatever clips were sampled that step.

### 2. MSE not cosine.

When target `z` is unit-normalized and prediction `ẑ` is not:

```
‖ẑ − z‖² = ‖ẑ‖² − 2(ẑ·z) + 1
```

The gradient pushes `ẑ` toward both the *direction* of `z` and toward the unit sphere.
Cosine loss only pushes toward direction. MSE does more work per step and provides
the BYOL-style collapse prevention via norm mismatch.

### 3. Normalized targets, un-normalized predictions.

- `zᵢ = normalize(f_θ̄(xᵢ))` — always unit norm, stop-gradient
- `z_comp = normalize(Σ zᵢ)` — always unit norm, stop-gradient
- `sₖ` (slots) — **not normalized**
- `ẑ_comp` — **not normalized**

This asymmetry is the collapse-prevention mechanism. Never normalize the prediction side.

### 4. No background latent in z_comp.

`z_comp = normalize(Σᵢ zᵢ)` sums only over labeled species `yᵢ ∈ Y(w)`.
No background term is added.

**Why:** Background features receive no L_fact gradient (no background species in the
label set). The unmatched slots learn background/noise only through L_comp and L_cls.
Background naturally fades from the factorized representation without explicit denoising.
Adding a background latent would require defining "what background sounds like" — which
is site-specific and poorly defined.

### 5. Slots are not species-specific. Assignment is arbitrary per step.

Hungarian re-assigns every step. Slot 1 in step 100 may match species A; in step 101
it may match species B. The classifier reads all K slots simultaneously — it is not
a per-slot species lookup.

This is the correct interpretation: slots are acoustic factorization slots, not species
memory cells. The classifier learns to aggregate species evidence across all K slots.

### 6. Training unit is the 5-second labeled soundscape window.

`train_soundscapes_labels.csv` gives species labels per 5-second window. That is the
training atom: one forward pass per labeled window. The window is the "context"; the
clips for its labeled species are the "targets."

This gives a clean 1:1 regression pair:
```
one window → one z_comp → one MSE target
```

### 7. K=8 slots.

BirdCLEF 2026 soundscape windows typically have 3–8 labeled species per window.
K=8 gives slack for background/noise to occupy extra slots without forcing species
into background slots. Hungarian matching handles K > N automatically.

Ablation plan: K = 4, 8, 12, 16 after initial runs at K=8.

### 8. EMA with τ=0.996.

```
θ̄ ← τ·θ̄ + (1−τ)·θ        updated every step, after gradient update
```

Target encoder is **never** updated by gradients. EMA keeps the target space stable
while still improving with training. τ=0.996 is the standard BYOL/I-JEPA value.

---

## Training Modes

### Mode 3 (Primary): Natural Labeled Soundscape

The primary training mode. Uses `train_soundscapes_labels.csv` directly.

```
soundscape window w with labels Y(w) = {y₁, …, yN}
  → f_θ → P1 slots → Clf → L_cls (BCE)
               ↓
    slots → Hungarian match → L_fact (vs {zᵢ})
    slots → P2 → ẑ_comp → L_comp (vs z_comp)

target path:
  for each yᵢ: sample xᵢ ~ train_audio/yᵢ/
  zᵢ = normalize(sg(f_θ̄(S(xᵢ))))
  z_comp = normalize(Σ zᵢ)
```

### Mode 2 (Synthetic Augmentation): Audio Mixing

Construct controlled synthetic examples where the exact component set is known.

```
x_mix = mix(x_A, x_B, x_bg)    (audio-domain mixing)
x_mix → f_θ → P1 → losses (same as Mode 3)
z_comp = normalize(z_A + z_B + z_bg)  from f_θ̄ of individual clips
```

Useful for curriculum: start with synthetic mixtures where the component set is exact,
then shift to natural soundscapes. Labels for synthetic mixtures are 1.0 (not soft 0.75).

### Mode 1 (Step 2): Masked JEPA Pretraining

Standard I-JEPA masked latent prediction on soundscape windows. No labels required.
Used to initialize `f_θ` before Step 3 fine-tuning.

Uses energy-guided masking: patches sampled with probability proportional to log-mel
energy, so high-energy patches (likely calls) land in context preferentially.

---

## Patch Sparsity Strategy

Most 5-second windows have acoustic content in only a fraction of patches. Handling
this correctly is important for all three training modes.

### Why JEPA is more robust than pixel reconstruction

MAE predicts raw spectrogram values for masked patches — background texture gets the
same gradient as a call. JEPA predicts in **latent space**: the target encoder naturally
collapses silent/noise patches into a compact region, so correctly predicting silence
gets near-zero gradient. The representation space does the denoising.

### Step 2 — Energy-guided masking

```python
energy = patch_logmel.mean(dim=(-1, -2))          # (N_patches,)
weights = torch.softmax(energy / temp, dim=0)
context_idx = torch.multinomial(weights, n_context, replacement=False)
```

### Step 3 — Window activity gate for L_fact / L_comp

```python
window_activity = logmel.max().item()
fact_weight = min(1.0, max(0.0, (window_activity - silence_db) / 20.0))
L = L_cls + fact_weight * (λ₁ * L_fact + λ₂ * L_comp)
```

Fully silent windows contribute only L_cls (all-zero labels = correct negative examples).
Do not filter silent windows entirely — the model needs negative examples.

### Slot attention handles patch sparsity naturally

P1's cross-attention slots compete for patches via softmax. In sparse windows,
background patches group into one or two background slots. No special treatment needed.

---

## Slot Attention (P1): Implementation

```python
# dots: (B, K, N) — each slot attends to all patches
dots = torch.einsum('bkd,bnd->bkn', q, k) * scale

# softmax over K: each patch assigns probability across slots
attn = dots.softmax(dim=1)           # (B, K, N)

# normalize over N: weighted average of patches per slot
attn_n = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)

updates = torch.einsum('bkn,bnd->bkd', attn_n, v)   # (B, K, D)

# GRU recurrence for 3 iterations
slots = gru(updates.reshape(B*K, D), prev.reshape(B*K, D)).reshape(B, K, D)
```

Three attention iterations. Slots initialized from learned mu/sigma (not from patches).

---

## Hyperparameters (Settled)

| Parameter | Value | Rationale |
|---|---|---|
| K (slots) | 8 | > typical 3–5 species per window; ablate 4, 12, 16 |
| D (latent dim) | 128 | small enough for fast iteration |
| τ (EMA momentum) | 0.996 | standard BYOL/JEPA |
| λ₁ (L_fact weight) | 0.5 | equal to λ₂ initially |
| λ₂ (L_comp weight) | 0.5 | |
| Patch size | 16×16 | on 64-mel × 512-frame spectrogram |
| N_patches | 128 | (64/16) × (512/16) |
| Attention iterations | 3 | standard Slot Attention |
| Learning rate | 1e-3 (AdamW) | cosine decay with 2-epoch warmup |
| Batch size | 64 (T4) / 128 (A100) | auto-scaled by GPU capability |

---

## GPU Precision Auto-Selection

```python
cap = torch.cuda.get_device_capability()
cap_val = cap[0] + cap[1] / 10

if cap_val >= 8.0:    amp_dtype = torch.bfloat16   # A100, H100
elif cap_val >= 7.0:  amp_dtype = torch.float16    # T4, V100
else:                 amp_dtype = torch.float32    # P100 (sm_60)

use_amp = (amp_dtype != torch.float32)
scaler  = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
```

P100 (sm_60) requires `torch==2.2.2+cu118` — torch ≥ 2.3 dropped sm_60 support.
Reinstall guard runs at kernel startup when cap_val < 7.0.

---

## Training Order

| Step | Objective | Data |
|---:|---|---|
| **1** | Supervised BCE baseline | Curated clips + labeled soundscape windows |
| 2 | JEPA pretraining (masked latent, energy-guided) | All soundscape windows |
| **3** | Factorize-and-Compose fine-tune | Labeled windows + curated clips |
| 4 | Pseudo-label expansion | High-confidence unlabeled windows |
| 5 | Threshold calibration per species | Val soundscape windows |

Steps 1 and 3 are the core. Step 2 gives a better encoder initialization for Step 3.

---

## Forward Pass Output (Real Audio, Random Init)

Demonstrated on actual BirdCLEF 2026 audio:
- Soundscape: `BC2026_Train_0019_S22_20211104_200000.ogg`, window [0:5s]
- Species A: `22973/iNat28425.ogg` (Whistling Grass Frog)
- Species B: `65380/iNat1650638.ogg` (Dwarf Tree Frog)

```
cos(z_A, z_B)    = +0.6209   (species are distinct even at random init)
cos(z_A, z_comp) = +0.9002   (z_comp lies between z_A and z_B — correct)
cos(z_B, z_comp) = +0.9002

Hungarian assignment (random init):
  slot 1 → 22973 Whistling Grass Frog   1-cos = 1.03
  slot 0 → 65380 Dwarf Tree Frog        1-cos = 0.94
  slots 2, 3 → unmatched (background)

L_cls  = 0.7345
L_fact = 0.3647   (sum over matched slots)
L_comp = 0.0472
L_total = 0.9405

f_θ  gradient norm: 0.007168  (gradients flow)
f_θ̄  gradient norm: 0.000000  (stop-gradient confirmed)
```

After training, matched 1-cos values should drop toward 0.

---

## What Remains Open

| Question | Status |
|---|---|
| Optimal K | Ablate 4, 8, 12, 16 in Step 3 |
| λ₁, λ₂ sweep | Start 0.5 / 0.5, ablate 0.1–1.0 |
| P2 architecture | MLP (current) vs attention pooling over slots |
| Clf architecture | Linear on flattened K×D (current) vs per-slot attention |
| Audio mixing for Mode 2 | Implement after Step 3 baseline exists |
| Slot diversity penalty | Add if slots collapse in practice |
| Energy-guided masking temp | Tune in Step 2 |
| Soundscape val split | First 10% of soundscape files by filename sort |

---

## What Is NOT Part of This Architecture

- **No species memory bank** — clips sampled fresh, not cached
- **No background prototype** in z_comp — implicit background suppression via unmatched slots
- **No contrastive loss** — all losses are regression toward specific vectors
- **No cosine loss** — MSE provides stronger gradient (direction + norm)
- **No species-specific slots** — slots are assignment-agnostic; Hungarian re-assigns every step
- **No normalized predictions** — only targets are normalized (the BYOL asymmetry)
- **No external pretrained weights** — scratch initialization throughout

---

## Further Reading

- [I-JEPA (Assran et al. 2023)](https://arxiv.org/abs/2301.08243) — latent prediction without reconstruction
- [Slot Attention (Locatello et al. 2020)](https://arxiv.org/abs/2006.15055) — learning object-centric representations
- [BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733) — collapse prevention via asymmetric targets
- [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) — competition page
