# Reading List — TLH-3: Applied Models: Diffusion Transformers + Video & World Models

> 30 papers across 5 themes. ⭐ = essential pre-session reading (read these 5 minimum).
> Estimated total: ~24 hrs. Pre-session essentials: ~4 hrs.

---

## Theme 1: Diffusion Transformers

*Replacing the U-Net backbone with a Vision Transformer.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 1 | **Scalable Diffusion Models with Transformers (DiT)** | Peebles & Xie | ICCV 2023 | The paper. Patchify → ViT blocks → adaLN → FID power law. ~1.5 hrs |
| 2 | **SiT: Scalable Interpolant Transformers** | Ma, Goldstein et al. | 2024 | DiT + stochastic interpolants (flow matching) → state-of-the-art FID |
| 3 | **An Image is Worth 16x16 Words (ViT)** | Dosovitskiy et al. | ICLR 2021 | The foundational ViT paper — understand patchification and positional encoding |
| 4 | **U-Net: Convolutional Networks for Biomedical Image Segmentation** | Ronneberger et al. | MICCAI 2015 | The U-Net paper — understand what DiT replaces and why |
| 5 | **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** | Podell et al. | ICLR 2024 | Scaled U-Net baseline — compare to DiT scaling |

---

## Theme 2: Video Diffusion Models

*Extending spatial diffusion to the temporal dimension.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 6 | **Video Generation Models as World Simulators (Sora)** | Brooks et al. (OpenAI) | 2024 | The technical report. Spacetime patches, variable resolution/duration, physics emergence. ~1 hr |
| 7 | **Video Diffusion Models** | Ho et al. | NeurIPS 2022 | The foundational video diffusion paper — factorized space-time attention |
| 8 | **Stable Video Diffusion** | Blattmann et al. | 2023 | SVD: fine-tune image LDM with temporal layers — production video generation |
| 9 | **Make-A-Video: Text-to-Video Generation without Text-Video Data** | Singer et al. | ICLR 2023 | Leverage image text-to-image models for video — spatial/temporal factorization |
| 10 | **CogVideo: Large-scale Pretraining for Text-to-Video Generation** | Hong et al. | ICLR 2023 | Chinese video generation model — different architecture choice |
| 11 | **Imagen Video: High Definition Video Generation with Diffusion Models** | Ho et al. (Google) | 2022 | Cascaded video diffusion — spatial then temporal super-resolution |

---

## Theme 3: World Models and Diffusion

*Diffusion as a predictive model of environment dynamics.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 12 | **DIAMOND: Diffusion for World Model** | Alonso et al. | NeurIPS 2024 | Diffusion model for RL — generates next observation given action. The key paper. |
| 13 | **Dreaming with Transformers (TWM)** | Robine et al. | 2023 | Transformer world model for Atari — baseline for DIAMOND |
| 14 | **DreamerV3: Mastering Diverse Domains through World Models** | Hafner et al. | NeurIPS 2023 | The non-diffusion baseline — RSSM + straight-through gradients |
| 15 | **Genie: Generative Interactive Environments** | Bruce et al. (DeepMind) | 2024 | Foundation world model — generate interactive environments from image |
| 16 | **UniSim: Learning Interactive Real-World Simulators** | Yang et al. | ICLR 2024 | Real-world simulator conditioned on text/action via diffusion |
| 17 | **Pandora: Towards General World Model with Natural Language Actions** | Xiang et al. | 2024 | LLM-guided diffusion world model |

---

## Theme 4: Scaling Laws and Architecture Analysis

*Understanding why ViTs scale and U-Nets don't.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 18 | **Scaling Laws for Neural Language Models** | Kaplan et al. (OpenAI) | 2020 | The original scaling laws — helps understand why DiT follows same trends |
| 19 | **Scaling Vision Transformers** | Zhai et al. (Google Brain) | CVPR 2022 | ViT scaling behavior — directly informs DiT design choices |
| 20 | **Training Compute-Optimal Large Language Models (Chinchilla)** | Hoffmann et al. | NeurIPS 2022 | Optimal compute allocation — informative for DiT training budget |
| 21 | **eDiff-I: Text-to-Image Diffusion Models with Ensemble of Expert Denoisers** | Balaji et al. | ECCV 2022 | Specialist U-Nets for different noise levels — alternative to single DiT |
| 22 | **Elucidating the Design Space of Diffusion-Based Generative Models (EDM)** | Karras et al. (NVIDIA) | NeurIPS 2022 | Systematic architecture + schedule analysis — high-quality baselines |

---

## Theme 5: Multimodal and Long-Form Generation

*Video generation, 3D, and beyond.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 23 | **CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer** | Yang et al. | 2024 | Expert DiT for video — production-scale open model |
| 24 | **Open-Sora: Democratizing Efficient Video Production for All** | — | 2024 | Open-source Sora reproduction — implementation reference |
| 25 | **Latte: Latent Diffusion Transformer for Video Generation** | Ma et al. | 2024 | Latent video DiT — efficient implementation |
| 26 | **3D-DiT: Scalable Video Diffusion Transformer** | — | 2024 | True 3D patchification for video |
| 27 | **Movie Gen: A Cast of Media Foundation Models** | Polyak et al. (Meta) | 2024 | Meta's production video/audio/image model suite |
| 28 | **Wan: Open and Advanced Large-Scale Video Generative Models** | — (Alibaba) | 2025 | State-of-the-art open video generation |
| 29 | **World Models Survey: Towards Autonomous AI Agents** | Zhu et al. | 2024 | Survey of world modeling approaches — situates diffusion world models |
| 30 | **Diffusion Models for Video Prediction and Infilling** | Höppe et al. | 2022 | Early work on video infilling — historical context |

---

## Suggested Reading Order

**Before the session (~4 hrs, ⭐ only):**
1. **DiT** (#1) — 90 min — architecture, adaLN, FID scaling curves — the core paper
2. **Sora** (#6) — 60 min — spacetime patches, variable resolution, emergent physics
3. **DIAMOND** (#12) — 60 min — diffusion world model for RL — the most novel application
4. **Video Diffusion Models** (#7) — 30 min — factorized attention, the baseline

**To go deeper:**
- Architecture: ViT (#3) → DiT (#1) → SiT (#2) for the progression
- Video scaling: Make-A-Video (#9) → Stable Video Diffusion (#8) → Sora (#6)
- World models: DreamerV3 (#14) → DIAMOND (#12) → Genie (#15) → UniSim (#16)

**TLH-4 prep:**
- Flow Matching on General Geometries (TLH-2 reading #8) — the bridge to biological manifolds
