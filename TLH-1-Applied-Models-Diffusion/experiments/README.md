# Experiments — TLH-1: Diffusion Models from Scratch

> Two notebooks. No GPU required. MNIST trains in ~5 minutes on CPU.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Demos

### Demo 01 — DDPM on MNIST from Scratch
**File:** `demo_01_ddpm_mnist.ipynb`

What it covers:
- Noise schedule (linear β schedule)
- Forward process: closed-form `q(x_t | x_0)` — jump to any timestep directly
- U-Net architecture: Conv + ResNet blocks + sinusoidal time embedding
- Training loop: denoising objective (MSE on predicted noise)
- DDPM sampling: 1000-step reverse process
- Visualization: noise trajectory at t = 0, 250, 500, 750, 1000

Expected output: generated MNIST digits after 10 epochs of training.

---

### Demo 02 — DDIM Fast Sampling + Noise Schedule Ablation
**File:** `demo_02_ddim_sampling.ipynb`

What it covers:
- DDIM deterministic ODE sampling: same model, 50 steps instead of 1000
- Speed comparison: DDPM (1000 steps) vs DDIM (50 steps) vs DDIM (10 steps)
- Noise schedule ablation: linear vs cosine schedule effects
- Latent space interpolation: walk between two noise seeds

Expected output: side-by-side generated digits + timing comparison + loss curves for both schedules.

---

## Configuration

Both notebooks use these shared hyperparameters:

```python
T          = 1000     # diffusion timesteps
beta_start = 1e-4     # noise schedule start
beta_end   = 0.02     # noise schedule end  
img_size   = 28       # MNIST image size
batch_size = 128
lr         = 2e-4
epochs     = 10       # enough for clear digits
model_dim  = 64       # U-Net channel base
```

## Hardware

Everything runs on CPU. Approximate runtimes:
- Training 10 epochs: ~5 minutes on CPU
- DDPM 1000-step sampling (batch of 16): ~45 seconds
- DDIM 50-step sampling (batch of 16): ~2.5 seconds
