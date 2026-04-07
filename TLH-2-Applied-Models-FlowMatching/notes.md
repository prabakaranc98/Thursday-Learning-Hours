# Session Notes — TLH-2: Applied Models: Flow Matching

> **Date:** Apr 17, 2026 · **Format:** Foundations → Frontiers → Demo → Discussion

---

## 🧠 Core Thesis

> *"Diffusion is one specific choice of probability path from data to noise — a curved, Gaussian path requiring a learned reversal. Flow matching chooses the simplest possible path: a straight line. Train a network to predict the velocity along that line. Sample by integrating the ODE with Euler steps. No SDE, no noise schedule, no ELBO. Just a velocity field."*

---

## 🏛️ FOUNDATIONS

### Section 1 — What Is a Probability Path?

Both diffusion and flow matching work by constructing a **time-indexed family of distributions**:

$$p_t(\mathbf{x}), \quad t \in [0, 1]$$

such that $p_0 = p_\text{data}$ and $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ (or any simple base).

**Diffusion (DDPM):** The path is determined by adding Gaussian noise:
$$p_t(\mathbf{x}) = \int q(\mathbf{x}|\mathbf{x}_0)\,p_\text{data}(\mathbf{x}_0)\,d\mathbf{x}_0, \quad q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$$

**Flow matching:** The path is determined by a vector field $v_t(\mathbf{x})$:
$$\frac{d\mathbf{x}}{dt} = v_t(\mathbf{x}), \quad \mathbf{x}_0 \sim p_\text{data}, \quad \mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Any valid velocity field gives a valid path. The question is: which is easiest to learn?

---

### Section 2 — Rectified Flow (Liu et al. 2022)

The simplest choice: **linear interpolation** between data and noise.

$$\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The velocity along this path is **constant**:
$$v_t(\mathbf{x}_t) = \frac{d\mathbf{x}_t}{dt} = \boldsymbol{\epsilon} - \mathbf{x}_0$$

So we train a network $v_\theta(\mathbf{x}_t, t)$ to predict $\boldsymbol{\epsilon} - \mathbf{x}_0$:

$$\boxed{\mathcal{L}_\text{RF} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|(\boldsymbol{\epsilon} - \mathbf{x}_0) - v_\theta(\mathbf{x}_t, t)\|^2\right]}$$

**Sampling:** Integrate the ODE from $t=1$ to $t=0$:
$$\mathbf{x}_{t - \Delta t} = \mathbf{x}_t - \Delta t \cdot v_\theta(\mathbf{x}_t, t)$$

With large enough $\Delta t$ (few steps), this is a single Euler step.

```python
# Forward process — linear interpolation
def forward_sample(x0, t, noise):
    return (1 - t) * x0 + t * noise

# Training objective — predict velocity
def flow_loss(model, x0):
    t     = torch.rand(x0.shape[0], device=x0.device)[:, None, None, None]
    noise = torch.randn_like(x0)
    x_t   = forward_sample(x0, t, noise)
    v_target = noise - x0                     # constant velocity along the path
    v_pred   = model(x_t, t.squeeze())
    return F.mse_loss(v_pred, v_target)

# Sampling — Euler ODE integration
@torch.no_grad()
def sample_rf(model, x_T, n_steps=50):
    x = x_T
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = 1.0 - i * dt                       # go from t=1 to t=0
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        v = model(x, t_batch)
        x = x - dt * v
    return x
```

---

### Section 3 — Conditional Flow Matching (Lipman et al. 2022)

Rectified flow uses **sample pairs** $(x_0, \epsilon)$ — this is fine, but leads to curved *marginal* trajectories when aggregated over many pairs.

**CFM insight:** Instead of marginal paths, condition on a specific $x_0$:

$$p_t(\mathbf{x} | \mathbf{x}_0) = \mathcal{N}(\mu_t(\mathbf{x}_0), \sigma_t^2 \mathbf{I})$$

with conditional velocity:
$$u_t(\mathbf{x} | \mathbf{x}_0) = \frac{\dot\sigma_t}{\sigma_t}(\mathbf{x} - \mu_t(\mathbf{x}_0)) + \dot\mu_t(\mathbf{x}_0)$$

The training objective becomes:
$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}\!\left[\|u_t(\mathbf{x}_t | \mathbf{x}_0) - v_\theta(\mathbf{x}_t, t)\|^2\right]$$

For linear paths ($\mu_t = (1-t)\mathbf{x}_0$, $\sigma_t = t\sigma_\min + (1-t)\sigma_\max$), CFM recovers rectified flow with small Gaussian noise around each path.

**Key advantage:** CFM can use *optimal transport* paths — straight lines that don't cross, leading to simpler learned velocity fields.

---

### Section 4 — Connection to Diffusion (DDPM as a Special Case)

DDPM's denoising objective can be rewritten as flow matching with a specific path choice:

| Method | Path $\mathbf{x}_t$ | Velocity target | Sampling |
|---|---|---|---|
| DDPM | $\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ | $-\boldsymbol\epsilon / \sqrt{1-\bar\alpha_t}$ (score) | SDE, 1000 steps |
| DDIM | Same path | Same velocity | ODE, 50 steps |
| Rectified Flow | $(1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$ | $\boldsymbol\epsilon - \mathbf{x}_0$ | ODE, 1 step (ideal) |

The key difference: rectified flow paths are straighter → shorter geodesics → fewer Euler steps needed.

---

### Section 5 — Stochastic Interpolants (Albergo & Vanden-Eijnden 2023)

A unified framework: define:
$$\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \beta_t\,\boldsymbol\epsilon + \gamma_t\,\mathbf{z}$$

where $\alpha_t, \beta_t$ interpolate between data and noise, and $\gamma_t\mathbf{z}$ is optional stochasticity.

Special cases:
- $\alpha_t = 1-t$, $\beta_t = t$, $\gamma_t = 0$: rectified flow
- $\alpha_t = \sqrt{\bar\alpha_t}$, $\beta_t = \sqrt{1-\bar\alpha_t}$, $\gamma_t > 0$: DDPM

This unifies diffusion and flow matching under one theory.

---

## 🔭 FRONTIERS

### Section 6 — Stable Diffusion 3 and FLUX

Esser et al. (2024) — SD3 uses **flow matching** with a **multimodal diffusion transformer (MMDiT)**:

```
[Text tokens (CLIP+T5)] ─────→ cross-attn ─────┐
[Image latent patches (VAE)] → self-attn blocks ┘ → denoised latent → VAE decoder
                                ↑ time conditioning via adaLN
```

Key upgrades over SD 1.5/2.x:
1. **Flow matching** instead of DDPM — straight paths, fewer steps
2. **MM-DiT** instead of U-Net — separate transformer streams for text + image that interact via joint attention
3. **T5-XXL** text encoder in addition to CLIP — better text understanding

**FLUX.1** (Black Forest Labs, 2024): same flow matching + DiT foundation, open weights.

---

### Section 7 — Reflow: Straightening Trajectories

Liu et al. show that flow matching paths can be iteratively straightened via **reflow**:

1. Train initial model $v_\theta^{(0)}$ on $(x_0, \epsilon)$ pairs
2. Generate new pairs by sampling: $x_0' = \mathrm{ODE}[\epsilon \to x]$, pair with $\epsilon$
3. Train $v_\theta^{(1)}$ on these re-paired $(x_0', \epsilon)$ — paths are now straighter
4. Repeat → paths converge to straight lines → **one-step sampling** becomes feasible

After 2-3 reflow rounds, single-step generation is possible without distillation.

---

### Section 8 — SiT: Scalable Interpolant Transformers

Ma et al. (2024) — Apply stochastic interpolant framework to DiT architecture:

- Replace DDPM noise with flow matching interpolants
- Show DiT + FM outperforms DiT + DDPM at same model size
- Demonstrates that flow matching is strictly better as a training objective for transformer-based denoisers

**Implication for TLH-3:** DiT + Flow Matching = the combination that powers frontier models.

---

## 🛠️ FRAMEWORK: RECTIFIED FLOW FROM SCRATCH

### Architecture

Same U-Net as TLH-1 — the velocity field network has the same signature as the noise predictor:
```
v_θ(x_t, t) : (B, C, H, W) × (B,) → (B, C, H, W)
```

The only changes from DDPM:
1. **No noise schedule** — $t$ is a uniform random variable in $[0, 1]$
2. **Forward process** is linear: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$
3. **Target** is $\boldsymbol\epsilon - \mathbf{x}_0$ instead of $\boldsymbol\epsilon$
4. **Sampling** is Euler integration, not DDPM reverse

```python
# Entire training difference from DDPM:
# DDPM:  noise_schedule → x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps  → predict eps
# RF:    t ~ Uniform[0,1] → x_t = (1-t)*x0 + t*eps            → predict (eps - x0)

def train_step_rf(model, x0, optimizer):
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)              # uniform in [0, 1]
    eps = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
    v_target = eps - x0                              # constant velocity
    v_pred = model(x_t, t)
    loss = F.mse_loss(v_pred, v_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

### Sampling Comparison

| Method | Steps | Formula |
|---|---|---|
| DDPM | 1000 | SDE with learned $\mu_\theta$ + noise |
| DDIM | 50 | ODE: predict $\hat x_0$ → re-noise |
| Rectified Flow | 50 | Euler: $x_{t-dt} = x_t - dt \cdot v_\theta(x_t, t)$ |
| RF (reflow) | 1 | Single Euler step (after reflow) |

---

## 💡 Key Takeaways

1. **Flow matching is strictly simpler than DDPM.** No noise schedule, no SDE, no ELBO. Just an ODE and a velocity field.

2. **Straight paths → faster sampling.** DDPM curves through data space; rectified flow goes in a straight line. Fewer Euler steps to integrate.

3. **The training objective is easier to optimize.** The target $\boldsymbol\epsilon - \mathbf{x}_0$ is the same at every timestep — no weighting scheme needed.

4. **DDIM is the bridge.** DDIM already views diffusion as an ODE — flow matching generalizes this to arbitrary path choices.

5. **SD3 and FLUX prove production viability.** Flow matching now powers the best open-weight text-to-image models.

---

## ❓ Discussion Questions

- Why do straighter paths require fewer ODE steps? What does this mean geometrically?
- CFM introduces conditioning on $x_0$ — how does this relate to classifier-free guidance?
- What is the optimal transport path, and why might it be better than naive linear interpolation?
- How does reflow relate to consistency model training? (Both straighten ODE trajectories)
- Can flow matching be applied to discrete data (text tokens)? What changes?
