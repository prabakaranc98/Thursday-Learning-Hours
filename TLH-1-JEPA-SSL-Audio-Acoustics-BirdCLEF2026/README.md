# TLH-1: Factorize-and-Compose Audio-JEPA

**Session:** Thursday Learning Hours · Session 1  
**Date:** May 2026  
**Case study:** BirdCLEF 2026 — wildlife species identification from passive acoustic monitoring

---

## The Problem

A passive recorder is left in the Pantanal wetlands. It records everything: frogs
calling, insects buzzing, wind, rain, and silence — all mixed together in one
continuous audio stream.

Every 5-second window of that stream can contain zero, one, or several species
calling at the same time. The task is to read each window and output a binary
vector over 234 species: *which of them are calling right now?*

Two datasets are available:

| Dataset | What it contains | Role |
|---|---|---|
| `train_soundscapes/` | Long passive recordings (messy, multi-species) | The real test distribution |
| `train_audio/` | Curated single-species clips (clean, labeled) | The "what each species sounds like" |
| `train_soundscapes_labels.csv` | Per-5s-window species labels for soundscapes | The supervision bridge |

The difficulty: soundscape labels are weak (the species may only call for 0.3s within
the 5s window), curated clips don't match soundscape conditions, and most species are
rare.

---

## Core Idea: Factorize-and-Compose

A soundscape window containing species A and B is, loosely, a *mixture* of two
acoustic sources. If we had a good representation space, the representation of
"A and B together" should be predictable from the representations of A and B
separately.

We use this as a training signal:

> **Learn to factorize a soundscape window into its constituent species latents,
> and verify the factorization is consistent with composing the individual species
> representations.**

This idea has three components:

1. **Factorize** — a slot-attention predictor (P1) splits the soundscape context
   into K independent slot vectors, each attending to a different part of the
   spectrogram.

2. **Compose** — a composition predictor (P2) takes all K slots and predicts what
   the "mixture latent" should look like.

3. **Supervise** — the target (what P2 should predict) is built from clean curated
   clips of the labeled species, encoded by a slow-moving EMA copy of the same
   encoder. No negatives, no contrastive loss — pure regression in latent space.

---

## Notation

```
w               5-second soundscape window (the context)
Y(w) = {A, B}  set of species labeled present in w
x_A, x_B       curated clips sampled from train_audio/ for each species

S(·)            LogMel spectrogram:  waveform → (1, F, T)
                F = 64 mel bins,  T ≈ 500 time frames

f_θ             context encoder   — trainable, gradients flow
f_θ̄            target encoder    — EMA copy of f_θ, stop-gradient always
                                    θ̄ ← τ·θ̄ + (1−τ)·θ,   τ = 0.996

h               patch token sequence: f_θ(S(w)) → (N_patches, D)
                N_patches = (F/ph) × (T/pw),  ph=pw=16,  D=128

P1              slot attention predictor
                h → K slot vectors,   K=4 fixed (K ≥ typical N per window)
s₁…sₖ          slot vectors: (K, D) — one slot per acoustic "source"
aₖ              attention weight map for slot k: which patches it attends to

P2              composition predictor
                {s₁…sₖ} → ẑ_comp ∈ R^D   (not normalised — MSE drives it)

Clf             classifier head
                {s₁…sₖ} → logits ∈ R^234  (sigmoid for multi-label)

z_A, z_B        target species latents  (unit norm, stop-gradient)
                z_i = normalize( f_θ̄(S(x_i)) )

z_comp          target composition (unit norm, stop-gradient)
                z_comp = normalize(z_A + z_B)
                         ↑ latent addition — NOT audio mixing

π*              Hungarian assignment: which slot matches which species latent
                π* = argmin_π  Σᵢ ‖normalize(s_{π(i)}) − zᵢ‖
```

---

## Forward Pass

Each training step processes one labeled soundscape window.

```
CONTEXT PATH  (gradients flow through everything here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

soundscape window  w
        │
        ▼
   S(w) = LogMel(w)          shape: (1, 64, 512)
        │                    64 mel bins × 512 time frames
        ▼
   h = f_θ(S(w))             shape: (128, 128)   [N_patches, D]
        │                    one embedding per 16×16 spectrogram patch
        ▼
   P1: slot attention         K=4 slots compete for 128 patches
        │
        ├──→  s₁, s₂, s₃, s₄    shape each: (128,)
        │       each slot focuses on a different region of the spectrogram
        │
        ├──→  P2(slots)  →  ẑ_comp    shape: (128,)
        │       predicted composition latent (un-normalised)
        │
        └──→  Clf(slots) →  logits    shape: (234,)
                species presence logits


TARGET PATH  (stop-gradient — no gradients ever flow here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

species clip  x_A  (Whistling Grass Frog)
        │
        ▼
   S(x_A) = LogMel(x_A)
        │
        ▼
   sg( f_θ̄(S(x_A)) )  →  pool patches  →  normalize  →  z_A    shape: (128,)

species clip  x_B  (Dwarf Tree Frog)  ──→  z_B    shape: (128,)

        z_comp = normalize(z_A + z_B)                            shape: (128,)
```

---

## Three Losses

All three are regression — no negatives, no contrastive pairs.

```
L_cls   =  BCE( logits,  multihot_Y(w) )
           ─────────────────────────────
           Standard multi-label classification.
           Teaches the classifier which species are present.
           Provides strong gradient from day one.

L_fact  =  (1/N) Σᵢ  MSE( s_{π*(i)},  zᵢ )
           ─────────────────────────────────
           Hungarian-matched slot-to-species regression.
           Pushes each matched slot toward the corresponding clean clip latent.
           Unmatched slots (K > N) receive no L_fact gradient — they learn
           background / silence through L_comp and L_cls only.

L_comp  =  MSE( ẑ_comp,  z_comp )
           ──────────────────────
           P2 must predict the composition of all species present.
           z_comp is unit-normalised; ẑ_comp is not.
           The norm mismatch (asymmetry) prevents representational collapse
           without any explicit constraint — same mechanism as BYOL.

L  =  L_cls  +  λ₁·L_fact  +  λ₂·L_comp       λ₁ = λ₂ = 0.5
```

**Why not contrastive?**  
Contrastive learning pushes representations apart using negatives. Here every
loss term is a regression toward a specific target vector. No negatives are
needed because the target is precise enough — predicting the exact direction
of z_comp in R^128 is its own hard task.

**Why MSE not cosine?**  
When the target z is unit-normalised and the prediction ẑ is not,
`‖ẑ − z‖² = ‖ẑ‖² − 2(ẑ·z) + 1`.
The gradient pushes ẑ toward both the *direction* of z and toward the unit
sphere. Cosine loss only pushes toward the direction. MSE does more work.

---

## Why the Target Composition Makes Sense

The key hypothesis: in a good representation space,

```
f_θ̄( "soundscape with A and B" )  ≈  normalize( f_θ̄(A) + f_θ̄(B) )
```

This is not assumed — it is *learned*. L_comp trains P2 to verify this
relationship. L_fact trains the slots so that the factorised representations
are consistent with the individual species embeddings.

As training progresses:
- L_cls differentiates species → z_A and z_B become more distinct
- L_fact pushes slots toward species-specific directions
- L_comp pushes the composition prediction to be consistent
- Backgrounds never reinforce any z_i target → background features fade
  from the factorised representation without any explicit denoising

The curated clips are the "ground truth sources." The soundscape labels are
the "who is in this mixture." Together they provide the regression target for
the latent factorization without requiring any audio source separation.

---

## Demo: Forward Pass on Real Audio

```bash
# Download audio samples from Kaggle (one-time)
kaggle competitions download birdclef-2026 \
  -f "train_soundscapes/BC2026_Train_0019_S22_20211104_200000.ogg" \
  -p /tmp/birdclef_sample/audio/soundscapes

kaggle competitions download birdclef-2026 \
  -f "train_audio/22973/iNat28425.ogg" \
  -p /tmp/birdclef_sample/audio/clips

kaggle competitions download birdclef-2026 \
  -f "train_audio/65380/iNat1650638.ogg" \
  -p /tmp/birdclef_sample/audio/clips

# Run forward pass demo
python experiments/step3_fc_jepa/forward_pass_demo.py
```

**What the demo shows (with real audio, random-init weights):**

```
Soundscape window  →  spec (1, 64, 512)  →  128 patches  →  4 slots
                                                                │
Species A clip     →  z_A  (128,)  unit norm                   │
Species B clip     →  z_B  (128,)  unit norm                   │
                   →  z_comp = norm(z_A + z_B)                 │
                                                                │
                              cos(z_A, z_B)    = 0.62  (distinct at init)
                              cos(z_A, z_comp) = 0.90  (z_comp between them)
                              cos(z_B, z_comp) = 0.90
```

Hungarian matching (random init — slots not yet specialised):
```
slot 1  →  22973 Whistling Grass Frog   1-cos = 1.03
slot 0  →  65380 Dwarf Tree Frog        1-cos = 0.94
slots 2, 3  →  unmatched  (background)
```

After training, matched 1-cos values should drop toward 0.

---

## Training Order

| Step | Objective | Data |
|---:|---|---|
| **1** | Supervised BCE baseline | Curated clips + labeled soundscape windows |
| 2 | JEPA pretraining (masked latent prediction) | All soundscape windows (unlabeled ok) |
| **3** | Factorize-and-Compose fine-tune | Labeled windows + curated clips |
| 4 | Pseudo-label expansion | High-confidence unlabeled windows |
| 5 | Threshold calibration | Val soundscape windows per species |

Steps 1 and 3 are the core contributions. Step 2 provides a better encoder
initialisation for Step 3.

---

## Files

```
experiments/
  step1_baseline/
    train.py               supervised BCE baseline (Kaggle-ready script)
    kernel-metadata.json   Kaggle kernel config
  step3_fc_jepa/
    forward_pass_demo.py   annotated single-step forward pass with real audio

artifacts/
  tikz/
    architecture_overview.tex   full model diagram (TikZ)
    mode2_composition.tex       training mode diagram (TikZ)
    geometry.tex                latent sphere geometry (TikZ)
    training_recipe.tex         pseudocode algorithms for Step 1 and Step 3
  pdf/   compiled PDFs
  png/   rasterised PNGs

notes.md        design decisions, data schema, patch sparsity strategy
```

---

## Further Reading

- [I-JEPA (Assran et al. 2023)](https://arxiv.org/abs/2301.08243) — latent prediction without reconstruction
- [Slot Attention (Locatello et al. 2020)](https://arxiv.org/abs/2006.15055) — learning object-centric representations
- [BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733) — collapse prevention via asymmetric targets
- [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) — competition page
