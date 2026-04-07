# TLH-3 · Applied Models: Diffusion Transformers + Video & World Models

> **Thursday Learning Hours — Session 3**
> *Replace the U-Net with a Vision Transformer. Scale to video. Simulate worlds.*

---

## 🎯 Motivation

DDPM and flow matching define the *objective* — but the *architecture* matters too. The original diffusion models used U-Nets borrowed from semantic segmentation. DiT (Peebles & Xie, 2023) showed you can replace the U-Net with a Vision Transformer and gain better scaling behavior. That shift enabled Sora.

This session covers the architectural leap from U-Net to ViT backbone, then extends to temporal generation (video diffusion), and finally to the most ambitious application: diffusion as a **world model** — a system that can simulate the future evolution of an environment.

**This session is Part 3 of the Applied Models arc:**
- **TLH-1:** DDPM — denoising objective, score matching
- **TLH-2:** Flow Matching — linear paths, ODE velocity
- **TLH-3 (this):** DiT — ViT backbone; Video — temporal diffusion; World models
- **TLH-4 (next):** Diffusion for Biology — proteins, molecules, MD

---

## 🗂️ Folder Structure

```
TLH-3-Applied-Models-DiT-Video/
├── README.md                              ← you are here
├── notes.md                               ← math + architecture diagrams
├── resources.md                           ← tools, implementations, datasets
├── links.md                               ← 80+ reference links
├── reading/README.md                      ← 30-paper annotated reading list
├── slides/SLIDES_OUTLINE.md              ← 19-slide presentation outline
├── tex/main.tex                           ← LaTeX technical report
└── experiments/
    ├── README.md                          ← experiment goals
    ├── demo_01_dit_mnist.ipynb            ← minimal DiT from scratch on MNIST
    ├── demo_02_video_frames.ipynb         ← temporal diffusion — sequential frames
    └── requirements.txt
```

---

## 🧠 Core Thesis

> *"The U-Net was a pragmatic choice — not a principled one. Vision Transformers process patches with global attention from the first layer, scale predictably with compute, and extend naturally to video by treating time as another patch dimension. DiT proved this works. Sora showed how far it scales."*

---

## 🔬 3F Structure

| Pillar | Coverage |
|---|---|
| 🏛️ **Foundations** | DiT architecture: patchify → ViT blocks with adaLN time/class conditioning → unpatchify; why ViT > U-Net for scaling |
| 🔭 **Frontiers** | Video diffusion (Stable Video Diffusion, Sora spacetime patches, temporal attention), world models (DIAMOND, DreamerV3 with diffusion), action-conditioned generation |
| 🛠️ **Frameworks** | Minimal DiT on MNIST with adaLN conditioning; temporal extension for digit sequences |

---

## 🚀 Quick Start

```bash
pip install -r experiments/requirements.txt

# Demo 1: DiT from scratch on MNIST with adaLN class conditioning
jupyter notebook experiments/demo_01_dit_mnist.ipynb

# Demo 2: Temporal frame generation — extend diffusion to sequences
jupyter notebook experiments/demo_02_video_frames.ipynb
```

---

## 📅 Session Info

- **Date:** Apr 24, 2026
- **Format:** 60-min (15 min Foundations · 15 min Frontiers · 20 min Live Demo · 10 min Discussion)
- **Pre-reads:** Peebles & Xie 2023 (DiT) + Brooks et al. 2024 (Sora) in `reading/README.md`
- **Audience:** Attendees of TLH-1/2 or anyone who understands the denoising objective

---

*Part of the [Thursday Learning Hours](../README.md) initiative — Applied Models track. Part 3 of 3.*
