# TLH-2 · Applied Models: Flow Matching

> **Thursday Learning Hours — Session 2**
> *Straight paths, simpler math — the ODE framework that powers Stable Diffusion 3 and FLUX.*

---

## 🎯 Motivation

Diffusion models use Gaussian noise as the forward process — a curved, expensive path from data to noise. Flow matching asks a cleaner question: **what if the path were a straight line?**

The answer is rectified flows and conditional flow matching — a framework with simpler math (no SDE, no noise schedule, no ELBO), faster training, fewer sampling steps, and identical (or better) quality. It's the framework behind Stable Diffusion 3, FLUX.1, and Meta's Voicebox.

**This session is Part 2 of the Applied Models arc:**
- **TLH-1 (prev):** DDPM — Gaussian forward, SDE, denoising objective, score matching
- **TLH-2 (this):** Flow matching — linear paths, ODE velocity field, simpler training
- **TLH-3 (next):** Diffusion Transformers — replace U-Net backbone with ViT, scale to video

---

## 🗂️ Folder Structure

```
TLH-2-Applied-Models-FlowMatching/
├── README.md                              ← you are here
├── notes.md                               ← math derivations + architecture diagrams
├── resources.md                           ← curated tools, papers, datasets
├── links.md                               ← 80+ reference links
├── reading/README.md                      ← 30-paper annotated reading list
├── slides/SLIDES_OUTLINE.md              ← 19-slide presentation outline
├── tex/main.tex                           ← LaTeX technical report
└── experiments/
    ├── README.md                          ← experiment goals
    ├── demo_01_rectified_flow_mnist.ipynb ← rectified flow from scratch on MNIST
    ├── demo_02_cfm_trajectories.ipynb     ← CFM, trajectory visualisation, reflow
    └── requirements.txt
```

---

## 🧠 Core Thesis

> *"Diffusion is one specific choice of probability path — curved, noisy, requiring 1000 steps to invert. Flow matching lets you choose any path. The simplest choice is a straight line: x_t = (1-t)·x_0 + t·ε. Train a network to predict the velocity (ε - x_0). Sample with a single Euler step. That's the entire framework."*

---

## 🔬 3F Structure

| Pillar | Coverage |
|---|---|
| 🏛️ **Foundations** | Probability paths, ODE velocity fields, conditional flow matching (CFM), connection to score matching and DDIM |
| 🔭 **Frontiers** | Stable Diffusion 3 / FLUX architecture, SiT (scalable interpolant transformers), consistency flow matching, audio/video flow models |
| 🛠️ **Frameworks** | Implement rectified flow from scratch on MNIST; compare trajectories with DDPM; visualise straightening via reflow |

---

## 🚀 Quick Start

```bash
pip install -r experiments/requirements.txt

# Demo 1: Rectified flow on MNIST (~3 min on CPU — simpler than DDPM)
jupyter notebook experiments/demo_01_rectified_flow_mnist.ipynb

# Demo 2: CFM + trajectory visualisation + reflow
jupyter notebook experiments/demo_02_cfm_trajectories.ipynb
```

> Everything runs on CPU. Flow matching trains faster than DDPM — fewer steps needed.

---

## 📅 Session Info

- **Date:** Apr 17, 2026
- **Format:** 60-min (15 min Foundations · 15 min Frontiers · 20 min Live Demo · 10 min Discussion)
- **Pre-reads:** Lipman et al. 2022 (Flow Matching) + Liu et al. 2022 (Rectified Flow) in `reading/README.md`
- **Audience:** Attendees of TLH-1 or anyone familiar with basic diffusion concepts

---

*Part of the [Thursday Learning Hours](../README.md) initiative — Applied Models track. Part 2 of 3.*
