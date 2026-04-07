# TLH-1 · Applied Models: Diffusion Models from Scratch

> **Thursday Learning Hours — Session 1**
> *The math, the intuition, and the code — implement DDPM on MNIST in one session.*

---

## 🎯 Motivation

Diffusion models are the backbone of modern generative AI: Stable Diffusion, DALL·E 3, Sora, RFdiffusion for proteins, molecular dynamics simulators. Yet most practitioners use them as black boxes. This session tears them open.

We go from first principles — what is a score function, why does the reverse process work, how does ELBO connect to denoising — to a working MNIST DDPM implementation that runs on your laptop in under 10 minutes. No GPUs required.

**This session is Part 1 of a two-part arc:**
- **TLH-1 (this):** Diffusion models — DDPM, score matching, DDIM, classifier-free guidance
- **TLH-2 (next):** Flow matching — the modern generalization of diffusion; simpler math, faster sampling

---

## 🗂️ Folder Structure

```
TLH-1-Applied-Models-Diffusion/
├── README.md                              ← you are here
├── notes.md                               ← math derivations + architecture diagrams
├── resources.md                           ← curated tools, papers, datasets
├── links.md                               ← 80+ reference links
├── reading/README.md                      ← 30-paper annotated reading list
├── slides/SLIDES_OUTLINE.md              ← 19-slide presentation outline
├── tex/main.tex                           ← LaTeX technical report
└── experiments/
    ├── README.md                          ← experiment goals
    ├── demo_01_ddpm_mnist.ipynb           ← DDPM from scratch on MNIST
    ├── demo_02_ddim_sampling.ipynb        ← fast DDIM sampling + visualization
    └── requirements.txt
```

---

## 🧠 Core Thesis

> *"A diffusion model is a learned reversal of a known destruction process. The forward process is fixed physics — Gaussian noise added step by step. The reverse is learned — a neural network that predicts 'what was here before the noise?' Score matching gives you the training objective. The rest is engineering."*

---

## 🔬 3F Structure

| Pillar | Coverage |
|---|---|
| 🏛️ **Foundations** | Forward/reverse SDE, score function, ELBO → denoising objective derivation, noise schedules, connection to VAEs |
| 🔭 **Frontiers** | DDIM (fast sampling), classifier-free guidance, consistency models, latent diffusion, Stable Diffusion architecture |
| 🛠️ **Frameworks** | HuggingFace `diffusers`, implement DDPM from scratch in PyTorch on MNIST; U-Net architecture hands-on |

---

## 🚀 Quick Start

```bash
pip install -r experiments/requirements.txt

# Demo 1: Full DDPM training from scratch on MNIST (~5 min on CPU)
jupyter notebook experiments/demo_01_ddpm_mnist.ipynb

# Demo 2: DDIM fast sampling + trajectory visualization
jupyter notebook experiments/demo_02_ddim_sampling.ipynb
```

> Everything runs on CPU. No GPU required. MNIST fits in RAM.

---

## 📅 Session Info

- **Date:** Apr 10, 2026
- **Format:** 60-min (15 min Foundations · 15 min Frontiers · 20 min Live Demo · 10 min Discussion)
- **Pre-reads:** Theme 1 in `reading/README.md` (Ho et al. 2020 + Song et al. 2021)
- **Audience:** Anyone who has used neural networks — no prior generative modeling required

---

*Part of the [Thursday Learning Hours](../README.md) initiative — Applied Models track. Part 1 of 2.*
