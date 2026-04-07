# Slides Outline — TLH-1: Diffusion Models from Scratch

> **19 slides · 60-minute session**
> Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)

---

## Slide 1 — Cover
**Title:** Diffusion Models from Scratch
**Subtitle:** The math, the intuition, and the code — DDPM on MNIST in one session
**Session:** TLH-1 · Thursday Learning Hours · Applied Models Track · Part 1 of 2
*Visual: Clean noise-to-image sequence — pure Gaussian noise → progressively cleaner MNIST digit*

---

## 🏛️ FOUNDATIONS (15 min, Slides 2–6)

## Slide 2 — Why Diffusion?
**Heading:** The Generative Modeling Landscape — Why Diffusion Won

| Model | Pros | Cons |
|---|---|---|
| GANs | Fast sampling | Training instability, mode collapse |
| VAEs | Stable training | Blurry outputs, posterior collapse |
| Normalizing Flows | Exact likelihood | Invertibility constraint limits architecture |
| **Diffusion** | **Best quality, stable, exact** | **Slow sampling (mitigated)** |

*"Diffusion beat GANs on FID in 2021 (Dhariwal & Nichol). Five years later it runs video."*

---

## Slide 3 — The Core Idea: Destroy Then Learn to Undo
**Heading:** Forward: Gradual Noise Addition · Reverse: Learned Denoising

*Visual: Two-row animation*
- Row 1 (forward): `[cat photo] → [slightly noisy] → ... → [pure noise]` — fixed, analytic
- Row 2 (reverse): `[pure noise] → [fuzzy cat] → ... → [sharp cat]` — learned by neural net

**Key insight:** The forward process is not learned — it's fixed Gaussian noise addition. Only the reverse process has parameters. This is what makes diffusion tractable.

---

## Slide 4 — The Math: Forward Process
**Heading:** $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I})$

- $T = 1000$ steps, noise schedule $\beta_1 < \cdots < \beta_T$
- $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ — cumulative noise
- **Jump directly to any $t$:** no need to run 1000 steps to get $\mathbf{x}_{500}$
- As $t \to T$: $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ — data destroyed

```python
x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
```
*Visual: Plot of $\bar{\alpha}_t$ vs $t$ (decaying from 1 to ~0) and sample images at t=0, 250, 500, 750, 1000*

---

## Slide 5 — The Loss: Predict the Noise
**Heading:** $\mathcal{L} = \mathbb{E}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$

1. Sample $\mathbf{x}_0$ from data, $t$ from $\{1,...,T\}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. Compute $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$
3. Predict $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ with a U-Net
4. Minimize MSE between true and predicted noise

**Why this works:** This is equivalent to score matching. The U-Net learns the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at every noise level $t$.

---

## Slide 6 — The Architecture: U-Net with Time Conditioning
**Heading:** The U-Net as a Denoiser

*Visual: U-Net diagram*
```
[Input: x_t + time t] → [DownBlocks] → [Bottleneck] → [UpBlocks + skips] → [ε_θ]
                                ↕ Self-attention at each scale
```

- **Time embedding:** sinusoidal → MLP → injected into every ResNet block via scale+shift
- **Skip connections:** preserve spatial detail (same as semantic segmentation U-Net)
- **Self-attention:** at 16×16 and 8×8 resolutions — global context

---

## 🔭 FRONTIERS (15 min, Slides 7–11)

## Slide 7 — DDIM: Deterministic Sampling in 50 Steps
**Heading:** From SDE to ODE — Same Model, 20× Faster

DDPM samples: stochastic SDE, 1000 steps
DDIM key insight: the diffusion process admits a **probability flow ODE** — fully deterministic

$$\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\cdot\hat{\mathbf{x}}_0(\mathbf{x}_t) + \sqrt{1-\bar\alpha_{t-1}}\cdot\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$$

- Same trained $\boldsymbol{\epsilon}_\theta$ — no retraining
- Choose any subset of $T$ timesteps: $[1000, 900, 800, ...]$ — skip steps
- 50 steps → virtually same quality. 10 steps → acceptable.

*Visual: Side-by-side samples: 1000 steps vs 50 steps vs 10 steps*

---

## Slide 8 — Classifier-Free Guidance: How Models Follow Prompts
**Heading:** CFG: Extrapolating Between Conditional and Unconditional

During training: randomly drop class/text label $y$ with probability $p$ → model learns both $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, y)$ and $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)$

During sampling:
$$\tilde{\boldsymbol{\epsilon}} = (1+w)\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, y) - w\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)$$

- $w=0$: unconditional  
- $w=7.5$: Stable Diffusion default — sharp, prompt-faithful  
- $w=30$: over-saturated, distorted  

*Visual: Grid of same prompt at w = 0, 2, 5, 10, 20 — quality vs diversity trade-off*

---

## Slide 9 — Latent Diffusion (Stable Diffusion)
**Heading:** Diffusion in Latent Space = Scale

*Visual: Architecture diagram*
```
Image (512×512×3) → [VAE Encoder] → z (64×64×4) → [DDIM] → z̃ → [VAE Decoder] → Image
                                                     ↑
                                              U-Net + cross-attention
                                              to CLIP text embeddings
```

- Encode to 4× smaller spatial latent → much cheaper diffusion
- Text conditioning via cross-attention in every transformer block
- Enables 512×512 generation on consumer GPUs

---

## Slide 10 — Consistency Models: One-Step Generation
**Heading:** From 1000 Steps → 1 Step

Key idea: learn $f_\theta(\mathbf{x}_t, t) = \mathbf{x}_0$ — predict clean image from any noisy input.

**Self-consistency constraint:**
$$f_\theta(\mathbf{x}_{t+1}, t+1) \approx f_{\hat\theta}(\mathbf{x}_t^{\Phi}, t)$$

where $\mathbf{x}_t^\Phi$ is one ODE step from $\mathbf{x}_{t+1}$, and $\hat\theta$ is an EMA of $\theta$.

Result: a single forward pass generates an image. Can refine with 2-4 steps for better quality.

---

## Slide 11 — The Landscape: Where Diffusion Is Going
**Heading:** DiT → Sora → RFdiffusion → TLH-3 and TLH-4

| Direction | What | TLH Session |
|---|---|---|
| **DiT** | Replace U-Net with Vision Transformer → scales to video | TLH-3 |
| **Video generation** | Sora, VideoLDM, CogVideo — temporal diffusion | TLH-3 |
| **World models** | Diffusion as predictive models of environment dynamics | TLH-3 |
| **Protein design** | RFdiffusion, FrameDiff — backbone coordinate diffusion | TLH-4 |
| **Molecular dynamics** | Diffusion over conformational ensembles | TLH-4 |
| **Flow matching (TLH-2)** | Straight probability paths, ODE formulation, faster | TLH-2 |

---

## 🛠️ DEMO (20 min, Slides 12–16)

## Slide 12 — What We Build
**Heading:** Demo Goal: DDPM from Scratch on MNIST — Runs on CPU in ~5 min

```
We build:
  ✓ Noise schedule (linear β schedule)
  ✓ Forward process (closed-form x_t sampling)
  ✓ U-Net (simplified: Conv + ResNet + time embedding)
  ✓ Training loop (denoising objective)
  ✓ Sampling loop (DDPM algorithm)
  ✓ DDIM fast sampling (50 steps vs 1000)
  ✓ Visualization: generation + noise trajectories
```

*No HuggingFace diffusers — pure PyTorch, ~200 lines of code*

---

## Slide 13 — Live Demo: Part 1 — The Math in Code
**Heading:** `demo_01_ddpm_mnist.ipynb` — Noise Schedule + Forward Process

```python
# 1. Noise schedule
betas = torch.linspace(1e-4, 0.02, T)          # linear schedule
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# 2. Forward process — q(x_t | x_0)
def q_sample(x0, t, noise):
    ab = alpha_bar[t][:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

# 3. Visualize corruption at t = 0, 250, 500, 750, 1000
```

*Show images: clean → increasingly noisy → pure Gaussian*

---

## Slide 14 — Live Demo: Part 2 — Training the U-Net
**Heading:** The Training Loop in 10 Lines

```python
for x0, _ in dataloader:
    t = torch.randint(0, T, (B,))          # random timestep
    noise = torch.randn_like(x0)           # random noise
    x_t = q_sample(x0, t, noise)           # add noise
    pred = model(x_t, t)                   # predict noise
    loss = F.mse_loss(pred, noise)         # denoising loss
    loss.backward()
    optimizer.step()
```

*Show: loss curve over training, watch it drop from ~1.0 to ~0.3*
*Show: samples at epoch 1, 3, 5, 10 — progressively better digits*

---

## Slide 15 — Live Demo: Part 3 — Sampling
**Heading:** DDPM vs DDIM Sampling

```python
# DDPM: 1000 steps
@torch.no_grad()
def p_sample_ddpm(model, x_T):
    x = x_T
    for t in reversed(range(T)):
        mu = (1/alphas[t].sqrt()) * (x - betas[t]/(1-alpha_bar[t]).sqrt() * model(x, t))
        if t > 0: x = mu + betas[t].sqrt() * torch.randn_like(x)
        else:     x = mu
    return x

# DDIM: 50 steps — same model, same quality
def p_sample_ddim(model, x_T, timesteps=50):
    ...
```

*Show: side-by-side generated MNIST digits — DDPM (1000 steps) vs DDIM (50 steps)*
*Timing: "DDPM = 45s on CPU, DDIM = 2.5s on CPU"*

---

## Slide 16 — Demo: Noise Schedule Ablation
**Heading:** Linear vs Cosine Schedule — What Happens?

*Visual: 2×2 grid*
- Linear schedule, DDPM
- Cosine schedule (Nichol & Dhariwal 2021), DDPM
- Linear schedule, training loss curve
- Cosine schedule, training loss curve

Key observation: cosine schedule spreads noise more uniformly across $t$, giving better training signal at low/high $t$ extremes.

---

## 🔄 DISCUSSION (10 min, Slides 17–19)

## Slide 17 — The Mathematical Zoo
**Heading:** Everything Is Connected

```
DDPM (Ho 2020)        Markovian SDE,    Gaussian forward, predict ε
DDIM (Song 2020)      Non-Markovian,    ODE view, deterministic, skip steps
Score-based (Song 2021)  Continuous SDE, Langevin dynamics, unified framework
Flow Matching (TLH-2) ODE, straight paths, simpler, no noise schedule needed
```

*The question for TLH-2: what if you chose a straighter path?*

---

## Slide 18 — Open Questions
**Heading:** What We Don't Know

1. What is the optimal noise schedule for a given data distribution?
2. Why does predicting $\boldsymbol\epsilon$ work better than $\mathbf{x}_0$ or $\mathbf{v}$ in some settings?
3. What is the right number of steps? (EDM framework argues ∞ continuous-time is optimal)
4. Can diffusion models learn causal structure, not just correlations? (CRL connection)
5. Is there a fundamental reason diffusion beats GANs, or is it engineering?

---

## Slide 19 — Takeaways + TLH-2 Preview
**Heading:** What to Take Away + What's Next

**Today:**
- Forward = fixed Gaussian corruption (analytic, closed-form)
- Reverse = learned denoising (U-Net, predicts $\boldsymbol\epsilon$)
- Loss = score matching = MSE on predicted noise
- DDIM = same model, ODE view, 20× faster
- CFG = prompt-following via conditional/unconditional interpolation

**TLH-2 — Flow Matching:**
- What if the forward process is a straight line instead of Gaussian diffusion?
- Flow matching: simpler math, faster training, same quality
- Implement rectified flows on MNIST (even simpler code than DDPM)
- The framework that powers Stable Diffusion 3 and FLUX

---

## Design Notes
- **Colors:** ivory `#f7f3ea`, copper `#8f3e22`, charcoal text
- **Code blocks:** IBM Plex Mono, pale cream background
- **Math:** typeset with KaTeX / LaTeX, copper for key equations
- **Diagrams:** thin-stroke, no fills — clean academic style
