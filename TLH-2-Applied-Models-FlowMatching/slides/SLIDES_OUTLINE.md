# Slides Outline — TLH-2: Flow Matching

> **19 slides · 60-minute session**
> Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)

---

## Slide 1 — Cover
**Title:** Flow Matching: Straight Paths, Simpler Math
**Subtitle:** From DDPM curves to straight lines — the ODE framework behind SD3 and FLUX
**Session:** TLH-2 · Thursday Learning Hours · Applied Models Track · Part 2 of 3
*Visual: Side-by-side trajectory comparison — curved diffusion paths vs straight flow matching paths converging to same data point*

---

## 🏛️ FOUNDATIONS (15 min, Slides 2–6)

## Slide 2 — Where We Left Off
**Heading:** DDPM was curved — can we do better?

DDPM recap (30 sec):
- Forward: $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ — curved, Gaussian path
- Training: predict noise $\boldsymbol\epsilon$
- Sampling: 1000-step SDE reverse (or DDIM ODE with 50 steps)

**The question:** Is there a simpler path?

*Visual: A single data point $\mathbf{x}_0$ and its DDPM trajectory to Gaussian noise — curved, not straight*

---

## Slide 3 — The Key Insight: A Straight Line
**Heading:** Rectified Flow — Linear Interpolation Between Data and Noise

$$\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol\epsilon, \quad t \in [0, 1]$$

- $t=0$: $\mathbf{x}_0$ — clean data
- $t=1$: $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ — pure noise
- The velocity along this path is **constant**: $v = \boldsymbol\epsilon - \mathbf{x}_0$

*Visual: Same data point — straight-line path from $\mathbf{x}_0$ to $\boldsymbol\epsilon$. No curvature.*

*"If the path is a straight line, you only need a ruler to walk it back."*

---

## Slide 4 — The Training Objective
**Heading:** $\mathcal{L} = \mathbb{E}\!\left[\|(\boldsymbol\epsilon - \mathbf{x}_0) - v_\theta(\mathbf{x}_t, t)\|^2\right]$

1. Sample $\mathbf{x}_0$ from data, $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $t \sim \mathrm{Uniform}[0, 1]$
2. Compute $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$
3. Predict velocity $v_\theta(\mathbf{x}_t, t)$
4. Minimize MSE against true velocity $\boldsymbol\epsilon - \mathbf{x}_0$

**Compared to DDPM:**

| | DDPM | Rectified Flow |
|---|---|---|
| Forward process | $\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ | $(1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$ |
| Target | $\boldsymbol\epsilon$ | $\boldsymbol\epsilon - \mathbf{x}_0$ |
| $t$ distribution | Uniform over $\{1,...,T\}$ | Uniform over $[0, 1]$ |
| Noise schedule | Required | Not needed |

---

## Slide 5 — Sampling: Euler Integration
**Heading:** ODE Sampling — No SDE, No Stochasticity

$$\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}_t, t) \implies \mathbf{x}_{t-\Delta t} = \mathbf{x}_t - \Delta t \cdot v_\theta(\mathbf{x}_t, t)$$

```python
# Euler sampling — 50 steps
x = torch.randn(N, C, H, W)          # start from noise at t=1
dt = 1.0 / n_steps
for i in range(n_steps):
    t = 1.0 - i * dt                  # t goes from 1 → 0
    v = model(x, t)
    x = x - dt * v                    # Euler step
```

Fewer steps work because the path is **straighter** — fewer function evaluations needed to integrate.

---

## Slide 6 — Conditional Flow Matching + Optimal Transport
**Heading:** CFM: Condition on x₀ for Better Paths

**Problem with naive RF:** marginal paths are still slightly curved (different $(x_0, \epsilon)$ pairs cross in data space).

**CFM solution:** condition the velocity on the specific $x_0$ being transported:
$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t, x_0, x_t \sim p_t(\cdot|x_0)}\!\left[\|u_t(x_t|x_0) - v_\theta(x_t, t)\|^2\right]$$

**Optimal Transport paths:** pair each $x_0$ with its nearest $\epsilon$ (minibatch OT approximation). Result: even straighter paths → fewer steps → better quality.

*Visual: Left — random pairing (crossing paths). Right — OT pairing (parallel, no crossing).*

---

## 🔭 FRONTIERS (15 min, Slides 7–11)

## Slide 7 — Reflow: Straightening Further
**Heading:** Iterative Path Straightening → 1-Step Generation

1. Train $v_\theta^{(0)}$ on $(x_0, \epsilon)$ pairs with random pairing
2. Generate new pairs: run $v_\theta^{(0)}$ to map $\epsilon \to \hat{x}_0$, store $(\hat{x}_0, \epsilon)$
3. Train $v_\theta^{(1)}$ on re-paired data — paths are now straighter
4. Repeat: after 2-3 rounds, paths are nearly straight → single Euler step works

*Visual: Trajectory curvature decreasing across reflow rounds. Right: 1-step samples after reflow.*

---

## Slide 8 — Stable Diffusion 3: MM-DiT + Rectified Flow
**Heading:** The Architecture Behind SD3

```
Text (CLIP + T5-XXL embeddings) ──→ Text tokens (N_T)
Image latent (VAE 64×64) ─────────→ Image tokens (N_I) patchify
                                     ↓
                            [MM-DiT block × L]
                            — text self-attention
                            — image self-attention
                            — text ↔ image cross-attention
                            — adaLN(t) for both streams
                                     ↓
                            Image tokens → unpatchify → VAE decode
```

**Why flow matching + MM-DiT:**
- Rectified flow → fewer sampling steps than DDPM LDM
- MM-DiT → joint text+image attention → better semantic alignment
- T5-XXL → rich text understanding beyond CLIP

---

## Slide 9 — FLUX.1: The Open-Weight Standard
**Heading:** Black Forest Labs — Open Flow Matching at Scale

**FLUX.1 variants:**
- `FLUX.1-dev` — full model, non-commercial, distilled for fast sampling
- `FLUX.1-schnell` — 4-step distilled, Apache 2.0
- `FLUX.1-pro` — API-only

**Key architectural feature:** "Hybrid DiT" — separate transformer streams for image + text (like MM-DiT) + additional parallel attention paths (from PaLI / Flamingo design).

**Training:** 12B+ parameters, flow matching, multi-aspect ratio training.

*Visual: FLUX sample grid — photorealistic diversity, strong prompt following.*

---

## Slide 10 — SiT: DiT + Flow Matching Combined
**Heading:** SiT — Scalable Interpolant Transformers

Ma et al. (2024): apply stochastic interpolant framework to the DiT backbone.

- Replace DDPM noise schedule with flow matching paths inside DiT
- Same architecture (adaLN, patchify, ViT blocks)
- FID 2.06 on ImageNet 256×256 — competitive with best diffusion models

**Lesson:** The path choice (diffusion vs FM) and the architecture (U-Net vs ViT) are **independent** decisions. Both matter. Best combination: FM + DiT.

---

## Slide 11 — Flow Matching Beyond Images
**Heading:** Voicebox, Proteins, Molecules — FM as a Universal Framework

| Domain | Paper | What flows | Manifold |
|---|---|---|---|
| **Text-to-speech** | Voicebox (Meta, 2023) | Mel spectrograms | $\mathbb{R}^{T \times F}$ |
| **Protein backbone** | FoldFlow (2023) | SO(3) frames | $SE(3)^N$ |
| **Molecules** | GeoBFN (2024) | Atom coords + types | $\mathbb{R}^{3N} \times \Delta^K$ |
| **Text** | Discrete FM (2024) | Token distributions | Probability simplex |

*"Flow matching is the new SDE — a universal training framework, not an image-specific technique."*

---

## 🛠️ DEMO (20 min, Slides 12–16)

## Slide 12 — What We Build
**Heading:** Demo Goal: Rectified Flow on MNIST — Simpler Than DDPM

```
We build:
  ✓ Linear forward process — x_t = (1-t)*x0 + t*eps (no noise schedule)
  ✓ Same U-Net architecture as TLH-1 (reuse your weights!)
  ✓ Training loop — 3 lines different from DDPM
  ✓ Euler ODE sampling — 50 steps
  ✓ Trajectory comparison: RF vs DDPM paths
  ✓ Reflow demo: watch paths straighten
```

*No noise schedule. No ELBO. No SDE. ~100 lines of code.*

---

## Slide 13 — Live Demo: Part 1 — The Forward Process
**Heading:** `demo_01_rectified_flow_mnist.ipynb` — Linear Paths

```python
# No noise schedule needed!
def q_sample_rf(x0, t, noise):
    t = t[:, None, None, None]     # broadcast
    return (1 - t) * x0 + t * noise

# Training objective
def rf_loss(x0):
    t     = torch.rand(B, device=x0.device)
    noise = torch.randn_like(x0)
    x_t   = q_sample_rf(x0, t, noise)
    v_target = noise - x0          # constant velocity
    return F.mse_loss(model(x_t, t), v_target)
```

*Show: forward trajectories at t = 0, 0.25, 0.5, 0.75, 1.0 — linear, not curved*

---

## Slide 14 — Live Demo: Part 2 — Sampling
**Heading:** Euler ODE — 50 Steps

```python
@torch.no_grad()
def sample_rf(model, n_samples=16, n_steps=50):
    x = torch.randn(n_samples, 1, 28, 28, device=device)   # start at t=1
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 - i * dt                   # t: 1 → 0
        t_batch = torch.full((n_samples,), t, device=device)
        v = model(x, t_batch)
        x = x - dt * v                     # Euler step
    return x
```

*Show: generated digits after 10 epochs — same quality as DDPM, trained faster*
*Timing: RF 50 steps = ~same speed as DDIM 50 steps*

---

## Slide 15 — Live Demo: Part 3 — Trajectory Comparison
**Heading:** RF vs DDPM — Why Straight Paths Win

*Visual: 2×2 panel*
- Top-left: DDPM trajectory in 2D embedding space — curved, meandering
- Top-right: RF trajectory — straight from noise to data
- Bottom-left: DDPM requires 50 steps for good quality
- Bottom-right: RF requires 10 steps for same quality (straighter path = fewer evaluations)

*"The path curvature determines the minimum number of steps. Straight paths = 1-step feasibility."*

---

## Slide 16 — Demo: Reflow — Watching Paths Straighten
**Heading:** Reflow: From Curved to Straight

```python
# Reflow: generate new (x0, eps) pairs with trained model
with torch.no_grad():
    eps = torch.randn(N, 1, 28, 28)
    x0_reflow = sample_rf(model_v0, eps.to(device))  # generate new clean images
    # Now re-pair: (x0_reflow[i], eps[i]) — same noise, different clean image
    # Train model_v1 on these straightened pairs
```

*Show: curvature metric across reflow iterations — approaches 0 (perfectly straight)*

---

## 🔄 DISCUSSION (10 min, Slides 17–19)

## Slide 17 — The Full Picture
**Heading:** From DDPM → DDIM → Flow Matching — One Trajectory

```
DDPM (Ho 2020)       SDE,  curved Gaussian paths,    1000 steps
DDIM (Song 2020)     ODE,  same curved paths,          50 steps (deterministic)
Rectified Flow       ODE,  linear paths,               50 steps (simpler training)
OT-CFM               ODE,  optimally straight paths,   10 steps
Reflow               ODE,  perfectly straight,           1 step (ideally)
```

*"Each step removed one source of complexity. Flow matching removes the last one: the noise schedule."*

---

## Slide 18 — Open Questions

1. What is the theoretically optimal coupling between $p_\text{data}$ and $p_\text{noise}$ for fastest sampling?
2. Can flow matching be applied to **discrete** data (text tokens) without continuous relaxation?
3. How does reflow relate to consistency model training? (Both straighten trajectories iteratively)
4. What happens when you combine OT-CFM with a DiT backbone at scale? (SiT gives a partial answer)
5. Is there a flow matching analogue of classifier-free guidance that doesn't double the forward passes?

---

## Slide 19 — Takeaways + TLH-3 Preview

**Today:**
- Forward = linear interpolation (no noise schedule, no ELBO)
- Velocity field $v_\theta = \epsilon - x_0$ — constant along each path
- Sampling = Euler ODE integration (same architecture, fewer steps)
- CFM + OT pairing → straighter paths → fewer steps
- SD3 / FLUX = FM + MM-DiT/hybrid DiT at scale

**TLH-3 — Diffusion Transformers + Video:**
- Replace U-Net with ViT backbone (DiT, adaLN conditioning)
- Extend spatial patches to spacetime patches → video
- Diffusion as a world model: predict next frame conditioned on action
- The architecture that enables Sora-class video generation

---

## Design Notes
- **Colors:** ivory `#f7f3ea`, copper `#8f3e22`, charcoal text
- **Code blocks:** IBM Plex Mono, pale cream background
- **Math:** typeset with KaTeX / LaTeX, copper for key equations
- **Trajectory visuals:** path diagrams in 2D with annotated waypoints
