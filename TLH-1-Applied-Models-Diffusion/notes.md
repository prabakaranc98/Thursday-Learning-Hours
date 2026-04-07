# Session Notes — TLH-1: Applied Models: Diffusion Models from Scratch

> **Date:** Apr 10, 2026 · **Format:** Foundations → Frontiers → Demo → Discussion

---

## 🧠 Core Thesis

> *"A diffusion model is a learned reversal of a known destruction process. The forward process is fixed, Markovian, and analytic. The reverse process is learned via a denoising objective that — by a beautiful identity — equals score matching. Understanding this identity is understanding diffusion."*

---

## 🏛️ FOUNDATIONS

### Section 1 — The Big Picture: What Is a Generative Model?

A generative model learns the data distribution $p_\text{data}(\mathbf{x})$ so we can sample from it.

**Three families of generative models:**
```
GANs:       implicit density  — sample by fooling a discriminator
VAEs:       explicit + approx — ELBO + reparameterization trick
Diffusion:  explicit + iterative — destroy then learn to reverse destruction
```

Diffusion models are currently dominant because:
- No adversarial instability (vs GANs)
- No posterior collapse (vs VAEs)
- Exact likelihood tractable (vs GANs)
- State-of-the-art sample quality on images, audio, molecules, video

---

### Section 2 — The Forward Process (Destroying Data)

The forward process $q$ gradually adds Gaussian noise over $T$ steps:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})$$

where $\beta_1 < \beta_2 < \cdots < \beta_T$ is the **noise schedule** (typically linear or cosine).

**Key trick — closed-form marginal:**
Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$\boxed{q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I})}$$

This means: **you can jump to any noise level in one step** — no need to iterate through all $T$ steps during training.

```python
# Sampling x_t from x_0 directly
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = extract(sqrt_alphas_bar, t, x0.shape)
    sqrt_one_minus = extract(sqrt_one_minus_alphas_bar, t, x0.shape)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise
```

**Intuition:** As $t \to T$, $\bar{\alpha}_t \to 0$ and $\mathbf{x}_t \to \mathcal{N}(\mathbf{0}, \mathbf{I})$. The data is destroyed into pure noise.

---

### Section 3 — The Reverse Process (Learning to Reconstruct)

The reverse process $p_\theta$ learns to undo the noise:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I})$$

The posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ is tractable (because both $q$'s are Gaussian):

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\; \tilde{\beta}_t \mathbf{I}\right)$$

where

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t$$

---

### Section 4 — The ELBO and the Denoising Objective

Training maximizes the variational lower bound (ELBO):

$$\mathcal{L} = \mathbb{E}_q\!\left[\underbrace{-\log p_\theta(\mathbf{x}_T)}_{\text{prior}} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t))}_{\text{denoising terms}} - \underbrace{\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}_{\text{reconstruction}}\right]$$

**The Ho et al. (2020) simplification:**
By parameterizing $\boldsymbol{\mu}_\theta$ to predict the noise $\boldsymbol{\epsilon}$ rather than $\mathbf{x}_0$, all KL terms simplify to:

$$\boxed{\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\underbrace{\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}}_{\mathbf{x}_t},\; t)\|^2\right]}$$

**In words:** Sample a random timestep $t$, add the corresponding amount of noise to get $\mathbf{x}_t$, and train the network to predict which noise was added.

```python
def p_losses(model, x0, t):
    noise = torch.randn_like(x0)
    x_noisy = q_sample(x0, t, noise)          # add noise
    predicted_noise = model(x_noisy, t)        # predict it back
    return F.mse_loss(predicted_noise, noise)  # simple L2
```

---

### Section 5 — Score Matching: The Deeper View

The **score function** of a distribution $p(\mathbf{x})$ is $\nabla_\mathbf{x} \log p(\mathbf{x})$.

Song & Ermon (2019) showed: if you train a network $s_\theta(\mathbf{x}, \sigma)$ to estimate $\nabla_\mathbf{x} \log p_\sigma(\mathbf{x})$ at noise level $\sigma$, you can generate samples via **Langevin dynamics**:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \epsilon\, s_\theta(\mathbf{x}_t, \sigma) + \sqrt{2\epsilon}\,\mathbf{z}_t$$

**The connection to DDPM:**
The noise prediction network $\boldsymbol{\epsilon}_\theta$ and the score estimator $s_\theta$ are related by:

$$s_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

They are the same model, just re-parameterized. The denoising loss **is** score matching.

```
Score function:  ∇_x log p(x)    →  direction "toward the data"
Denoising:       predict ε added  →  scaled negative score
```

---

### Section 6 — Sampling: The DDPM Algorithm

**Training (Algorithm 1):**
```
1. x0 ~ q(data)
2. t ~ Uniform({1,...,T})
3. ε ~ N(0, I)
4. x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
5. Gradient descent on ||ε - ε_θ(x_t, t)||²
```

**Sampling (Algorithm 2) — 1000 steps:**
```
1. x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   μ_θ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
   if t > 1: x_{t-1} = μ_θ + √β_t · z,   z ~ N(0, I)
   else:     x_0 = μ_θ
3. Return x_0
```

**The problem:** 1000 forward passes through the network to generate one sample. Slow.

---

## 🔭 FRONTIERS

### Section 7 — DDIM: Fast Sampling Without Retraining

Song et al. (2020) — Denoising Diffusion Implicit Models.

Key insight: the denoising objective doesn't require the forward process to be Markovian. You can define a non-Markovian forward process with the same marginals $q(\mathbf{x}_t \mid \mathbf{x}_0)$.

This gives a **deterministic ODE** for sampling:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\cdot\underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted }\mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

**Result:** Same trained model, but sample in 50 steps instead of 1000. **20× speedup.**

```python
# DDIM sampling step (eta=0 for deterministic)
x0_pred = (x_t - sqrt_one_minus_bar * eps_pred) / sqrt_bar
x_prev = sqrt_bar_prev * x0_pred + sqrt_one_minus_bar_prev * eps_pred
```

---

### Section 8 — Classifier-Free Guidance (CFG)

Ho & Salimans (2021). The trick that makes conditional diffusion models work.

**Classifier guidance (old):** Train an external classifier $p_\phi(y \mid \mathbf{x}_t)$, use its gradients to guide sampling:
$$\tilde{s}_\theta(\mathbf{x}_t, y) = s_\theta(\mathbf{x}_t) + \gamma \nabla_{\mathbf{x}_t} \log p_\phi(y \mid \mathbf{x}_t)$$

**Classifier-free (modern):** Train a single model that can be both conditional and unconditional by randomly dropping the conditioning label during training:
$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, y) = (1+w)\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, y) - w\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)$$

- $w=0$: unconditional
- $w=7.5$: typical Stable Diffusion default
- Higher $w$: sharper, more condition-faithful but less diverse

**This is why DALL·E 3 "follows your prompt" — CFG is the mechanism.**

---

### Section 9 — Latent Diffusion & Stable Diffusion

Rombach et al. (2022) — Running diffusion in the latent space of a VAE.

```
[Encoder E] → latent z = E(x)    (4× smaller spatial dims)
[Diffusion model] → operates on z, not x
[Decoder D] → x = D(z̃)
```

**Why:** Running 1000-step DDIM on 512×512 pixels = prohibitive. In 64×64 latent space = fast.

**U-Net with cross-attention** for text conditioning:
- Encoder: downsampling blocks with residual + attention
- Bottleneck: self-attention
- Decoder: upsampling blocks with skip connections from encoder
- Cross-attention to CLIP text embeddings at every resolution level

---

### Section 10 — Consistency Models

Song et al. (2023). **One-step generation.**

Key idea: learn a function $f_\theta(\mathbf{x}_t, t) \approx \mathbf{x}_0$ that is **self-consistent** — $f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t')$ for any $t, t'$ on the same ODE trajectory.

Training: enforce consistency between adjacent steps:
$$\mathcal{L} = d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}),\; f_{\hat{\theta}}(\mathbf{x}_{t_n}^{\Phi}, t_n))$$

where $\mathbf{x}_{t_n}^{\Phi}$ is one Euler step of the DDIM ODE, and $\hat{\theta}$ is a running average (EMA) of $\theta$.

**Result:** Single forward pass → sample. 100-1000× faster than DDPM.

---

## 🛠️ FRAMEWORK: DDPM FROM SCRATCH

### Architecture: The U-Net

```
Input: x_t ∈ ℝ^{H×W×C}, time embedding t ∈ ℝ^d

[Conv2d] → (H, W, 64)
  ↓ DownBlock(64→128)  + ResNet + SelfAttention
  ↓ DownBlock(128→256) + ResNet + SelfAttention
  ↓ [Bottleneck] ResNet + SelfAttention + ResNet
  ↑ UpBlock(256→128) + skip + ResNet + SelfAttention
  ↑ UpBlock(128→64)  + skip + ResNet + SelfAttention
[Conv2d] → (H, W, C)   ← predicted noise ε_θ(x_t, t)
```

**Time embedding:** Sinusoidal positional encoding (same as Transformer), then MLP → added to each ResNet block via learned scale+shift.

### Key Implementation Notes

```python
# Time embedding (sinusoidal)
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
    args = t[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)

# ResNet block with time conditioning
class ResBlock(nn.Module):
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]  # add time info
        h = self.conv2(self.norm2(h))
        return h + self.skip(x)  # residual
```

### MNIST-scale Configuration

```python
# Hyperparameters (runs on CPU in ~5 min)
T            = 1000          # diffusion steps
beta_start   = 1e-4          # noise schedule start
beta_end     = 0.02          # noise schedule end
img_size     = 28            # MNIST
batch_size   = 128
lr           = 2e-4
epochs       = 10            # enough to see results
model_dim    = 64            # U-Net channel multiplier
```

---

## 💡 Key Takeaways

1. **The forward process is fixed — only the reverse is learned.** This is what makes diffusion tractable: the forward destruction is analytic; only the neural network (reverse) has parameters.

2. **The denoising loss = score matching.** These are the same objective from different angles. This unification is the theoretical core of the field.

3. **DDIM shows the reverse process is an ODE, not an SDE.** The stochasticity in DDPM is optional — you can trade it for determinism and get 20× faster sampling with the same model.

4. **Classifier-free guidance is the mechanism of control.** Extrapolating between conditional and unconditional predictions is how you get prompt-following behavior.

5. **Latent diffusion = scale.** Running diffusion in pixel space is prohibitive at resolution. A VAE encoder/decoder + diffusion in latent space is how Stable Diffusion works.

6. **Flow matching (TLH-2) is the generalization.** DDPM is one specific choice of probability path. Flow matching lets you design the path — and simpler paths (straight lines) give faster sampling.

---

## ❓ Discussion Questions

- Why does predicting $\boldsymbol{\epsilon}$ work better than predicting $\mathbf{x}_0$ directly? (Hint: what does the loss gradient look like at high vs low $t$?)
- What is the difference between the SDE and ODE formulations of diffusion? When does each matter?
- Classifier-free guidance improves quality at the cost of diversity. What is the right trade-off for your application?
- How does diffusion compare to normalizing flows in terms of expressiveness, training complexity, and inference speed?
- What would a diffusion model over protein sequences look like? (Preview of TLH-4)

---

## 📝 DWS Prep Checklist

### DWS-A: Foundation Dive
- [ ] Read Ho et al. 2020 (DDPM) — the core paper (~1.5 hrs)
- [ ] Read Song et al. 2021 score-based SDE paper — understand the unification (~1 hr)
- [ ] Derive the closed-form marginal $q(\mathbf{x}_t \mid \mathbf{x}_0)$ from scratch on paper

### DWS-B: Frontier Exploration
- [ ] Read Song et al. 2020 (DDIM) — the fast sampling paper (~45 min)
- [ ] Read Ho & Salimans 2021 (CFG) — 6 pages, essential (~30 min)
- [ ] Skim Rombach et al. 2022 (LDM / Stable Diffusion) — understand the architecture

### DWS-C: Framework Build
- [ ] Run Demo 01 end-to-end — DDPM on MNIST training loop
- [ ] Run Demo 02 — DDIM sampling + compare speed vs quality curves
- [ ] Try changing the noise schedule (linear vs cosine) and observe effects
