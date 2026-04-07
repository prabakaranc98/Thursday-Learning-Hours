# TLH-4 · Applied Life: Diffusion for Biology & Molecular Dynamics

> **Thursday Learning Hours — Session 4**
> *From pixel generation to protein design — diffusion over SE(3) and conformational ensembles.*

---

## 🎯 Motivation

The diffusion framework is not limited to images. In structural biology, the "data" is the 3D geometry of proteins and molecules — positions and orientations in continuous space. The same denoising objective applies, but the geometry requires equivariance: the model must respect rotations and translations.

This session bridges the Applied Models arc (TLH-1–3) with the Applied Life arc (TLH-6). We cover:
- **Protein backbone diffusion** — RFdiffusion, FrameDiff: diffusion over SO(3) rotation frames
- **Small molecule generation** — EDM: E(3)-equivariant diffusion for 3D molecular graphs
- **Structure-based drug design** — DiffSBDD: conditional generation in protein pockets
- **Molecular dynamics** — MDGen: diffusion over conformational trajectories (preview of TLH-6)

---

## 🗂️ Folder Structure

```
TLH-4-Applied-Life-DiffusionBiology/
├── README.md                              ← you are here
├── notes.md                               ← math: SE(3) diffusion, equivariance, protein geometry
├── resources.md                           ← tools, datasets, pre-trained models
├── links.md                               ← 80+ reference links
├── reading/README.md                      ← 30-paper annotated reading list
├── slides/SLIDES_OUTLINE.md              ← 19-slide presentation outline
├── tex/main.tex                           ← LaTeX technical report
└── experiments/
    ├── README.md                          ← experiment goals
    ├── demo_01_molecular_diffusion_2d.ipynb   ← toy 2D point cloud diffusion (geometry intuition)
    ├── demo_02_backbone_diffusion_1d.ipynb    ← 1D torsion angle diffusion (protein analogue)
    └── requirements.txt
```

---

## 🧠 Core Thesis

> *"A protein backbone is a sequence of rigid frames — each frame is a rotation matrix R ∈ SO(3) and a translation t ∈ ℝ³. Diffusion over SE(3) = diffusion over these frames. The forward process adds noise on the rotation manifold (IGSO(3)) and in translation space (ℝ³ Gaussian). The reverse process is a neural network — usually an SE(3)-equivariant GNN or structure module — that learns to denoise frames back to a valid protein backbone. That is the entire basis of RFdiffusion."*

---

## 🔬 3F Structure

| Pillar | Coverage |
|---|---|
| 🏛️ **Foundations** | SE(3) geometry for proteins, IGSO(3) forward process, equivariant GNNs, the invariant point attention (IPA) module from AlphaFold 2 |
| 🔭 **Frontiers** | RFdiffusion (backbone design), FrameDiff, EDM (equivariant diffusion for molecules), DiffSBDD (pocket-conditioned drug design), MDGen (MD trajectories as diffusion) |
| 🛠️ **Frameworks** | Toy 2D molecular diffusion with distance constraints; 1D torsion angle diffusion as a protein analogue; ESM-IF interface |

---

## 🚀 Quick Start

```bash
pip install -r experiments/requirements.txt

# Demo 1: 2D point cloud diffusion — geometry and equivariance intuition
jupyter notebook experiments/demo_01_molecular_diffusion_2d.ipynb

# Demo 2: 1D torsion angle diffusion — protein backbone analogue
jupyter notebook experiments/demo_02_backbone_diffusion_1d.ipynb
```

---

## 📅 Session Info

- **Date:** May 1, 2026
- **Format:** 60-min (15 min Foundations · 15 min Frontiers · 20 min Live Demo · 10 min Discussion)
- **Pre-reads:** Watson et al. 2022 (RFdiffusion) + Hoogeboom et al. 2022 (EDM) in `reading/README.md`
- **Audience:** TLH-1/2/3 attendees — basic diffusion knowledge assumed; biology background helpful but not required

---

*Part of the [Thursday Learning Hours](../README.md) initiative — Applied Life track. Bridge between Applied Models (TLH-1–3) and Applied Life Survey (TLH-6).*
