# Reading List — TLH-1: Applied Models: Diffusion Models from Scratch

> 30 papers across 5 themes. ⭐ = essential pre-session reading (read these 5 minimum).
> Estimated total: ~25 hrs. Pre-session essentials: ~4 hrs.

---

## Theme 1: DDPM and the Denoising Objective

*The foundation — Ho et al. 2020 and the score-based unification.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 1 | **Denoising Diffusion Probabilistic Models (DDPM)** | Ho, Jain, Abbeel | NeurIPS 2020 | The paper. Closed-form marginals, denoising objective, MNIST/CIFAR results. ~1.5 hrs. |
| ⭐ 2 | **Score-Based Generative Modeling through SDEs** | Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole | ICLR 2021 | Unifies DDPM and score matching under a single SDE framework. The theoretical core. |
| 3 | **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** | Sohl-Dickstein et al. | ICML 2015 | Original diffusion paper — elegant but superseded; read for historical context |
| 4 | **Improved DDPM** | Nichol & Dhariwal | ICML 2021 | Learned noise schedule, importance sampling, better log-likelihood |
| 5 | **Elucidating the Design Space of Diffusion-Based Generative Models** | Karras et al. (NVIDIA) | NeurIPS 2022 | Systematic analysis of sampling schedules, preconditioning, and model architectures |

---

## Theme 2: Fast Sampling and ODE Solvers

*From 1000 steps to 10 — the DDIM revolution and beyond.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 6 | **Denoising Diffusion Implicit Models (DDIM)** | Song, Meng, Ermon | ICLR 2021 | Non-Markovian forward process → deterministic ODE → 20× speedup with same model |
| 7 | **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling** | Lu et al. | NeurIPS 2022 | Analytic form of linear part → fewer function evaluations; 10–20 NFE |
| 8 | **DPM-Solver++: Fast Solver for Guided Sampling** | Lu et al. | NeurIPS 2023 | Extends DPM-Solver to CFG setting |
| 9 | **PNDM: Pseudo Numerical Methods for Diffusion Models** | Liu et al. | ICLR 2022 | Higher-order ODE solvers for diffusion — historical importance |
| 10 | **Progressive Distillation for Fast Sampling** | Salimans & Ho | ICLR 2022 | Distill T-step DDIM into T/2-step student, repeat — path to 4-step sampling |

---

## Theme 3: Conditional Generation and Guidance

*Teaching diffusion models to follow instructions.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| ⭐ 11 | **Classifier-Free Diffusion Guidance** | Ho & Salimans | NeurIPS WS 2021 | The mechanism behind prompt-following — 6 pages, essential |
| 12 | **Diffusion Models Beat GANs on Image Synthesis** | Dhariwal & Nichol | NeurIPS 2021 | Classifier guidance + improved U-Net architecture |
| 13 | **DALL·E 2: Hierarchical Text-Conditional Image Generation** | Ramesh et al. | 2022 | CLIP embeddings + cascaded diffusion for text-to-image |
| 14 | **High-Resolution Image Synthesis with Latent Diffusion Models** | Rombach et al. | CVPR 2022 | Stable Diffusion's paper — VAE + cross-attention U-Net + LDM |
| 15 | **ControlNet: Adding Conditional Control to Diffusion Models** | Zhang et al. | ICCV 2023 | Fine-grained spatial control (edges, depth, pose) over pretrained models |

---

## Theme 4: Consistency and Distillation

*One-step generation and faster inference.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 16 | **Consistency Models** | Song et al. | ICML 2023 | Self-consistency training → single-step generation; the key distillation paper |
| 17 | **Improved Consistency Training** | Song & Dhariwal | NeurIPS 2023 | Improved training stability and FID |
| 18 | **Consistency Trajectory Models (CTM)** | Kim et al. | ICLR 2024 | Combines consistency + score matching; tunable quality/speed trade-off |
| 19 | **Score Distillation Sampling (SDS)** | Poole et al. (DreamFusion) | ICLR 2023 | Using 2D diffusion models as 3D scene priors via score distillation |
| 20 | **FlashDiffusion / LCM-LoRA** | Luo et al. | 2023 | Latent consistency models — 4-step SD with LoRA adapters |

---

## Theme 5: Applications and Extensions

*Diffusion beyond images — audio, molecules, video.*

| # | Paper | Authors | Year | Why Read |
|---|---|---|---|---|
| 21 | **WaveGrad: Estimating Gradients for Waveform Generation** | Chen et al. | ICLR 2021 | Audio waveform generation with diffusion |
| 22 | **DiffWave: A Versatile Diffusion Model for Audio Synthesis** | Kong et al. | ICLR 2021 | Parallel audio generation with DDPM |
| 23 | **Taming Transformers for High-Resolution Image Synthesis (VQGAN)** | Esser et al. | CVPR 2021 | VQ-VAE for latent space — the codebook approach vs. continuous latents |
| 24 | **DiT: Scalable Diffusion Models with Transformers** | Peebles & Xie | ICCV 2023 | Replace U-Net with Vision Transformer — scales better, enables Sora-class models |
| 25 | **SDXL: Improving Latent Diffusion Models for High-Resolution** | Podell et al. | ICLR 2024 | Scaled-up Stable Diffusion — architecture improvements and conditioning |
| 26 | **eDiff-I: Text-to-Image Diffusion with Ensemble of Expert Denoisers** | Balaji et al. | 2022 | Different U-Nets for different noise levels — specialization insight |
| 27 | **Cold Diffusion: Inverting Arbitrary Image Transforms** | Bansal et al. | NeurIPS 2023 | Generalized diffusion: blur, masking, downsampling as forward processes |
| 28 | **Soft Diffusion: Score Matching for General Corruptions** | Gao et al. | TMLR 2023 | Unified framework for non-Gaussian forward processes |
| 29 | **DiffSBDD: Structure-Based Drug Design with Diffusion** | Schneuing et al. | 2022 | Diffusion for 3D molecular generation conditioned on protein pocket |
| 30 | **MDGen: Simulation-Free Molecular Dynamics with Diffusion** | — | 2024 | Molecular dynamics trajectories as diffusion — preview of TLH-4 |

---

## Suggested Reading Order

**Before the session (~4 hrs, ⭐ only):**
1. **DDPM** (#1) — 90 min — core algorithm, derivations, read with pen and paper
2. **DDIM** (#6) — 45 min — understand the ODE view, the speedup
3. **Song et al. SDE** (#2) — 60 min — the unification; skim if short on time
4. **Classifier-Free Guidance** (#11) — 30 min — 6 pages, just read it
5. **Stable Diffusion / LDM** (#14) — 45 min — skim architecture section

**To go deeper:**
- Sampling efficiency: read #7 (DPM-Solver) then #16 (Consistency Models) for the arc
- Conditional control: read #12 (classifier guidance) → #11 (CFG) → #15 (ControlNet)
- Math: Song et al. SDE (#2) → Karras et al. EDM (#5) for full design space analysis

**For TLH-2 preview:**
- After this session, read #2 (Song SDE) Appendix on probability flow ODE — that's the bridge to flow matching
