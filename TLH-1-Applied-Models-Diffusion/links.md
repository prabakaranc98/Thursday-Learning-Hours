# Reference Links — TLH-1: Applied Models: Diffusion Models from Scratch

> 80+ links across papers, implementations, tutorials, and tools.

---

## Foundational Papers

### DDPM and Core Objective
- https://arxiv.org/abs/2006.11239 — Ho, Jain, Abbeel (2020) — DDPM
- https://arxiv.org/abs/2006.08205 — Song & Ermon (2020) — Improved Score Matching
- https://arxiv.org/abs/2011.13456 — Song et al. (2021) — Score-Based Generative Modeling through SDEs
- https://arxiv.org/abs/2102.09672 — Nichol & Dhariwal (2021) — Improved DDPM
- https://arxiv.org/abs/2206.00364 — Karras et al. (2022) — EDM: Elucidating the Design Space
- https://arxiv.org/abs/1503.03585 — Sohl-Dickstein et al. (2015) — Deep Unsupervised Learning using Nonequilibrium Thermodynamics

### Score Matching (Pre-DDPM)
- https://jmlr.org/papers/v6/hyvarinen05a.html — Hyvärinen (2005) — Estimation of Non-Normalized Statistical Models
- https://arxiv.org/abs/1907.05600 — Song & Ermon (2019) — Generative Modeling by Estimating Gradients of the Data Distribution (NCSN)
- https://arxiv.org/abs/2006.09011 — Song & Ermon (2020) — Improved Techniques for Training Score-Based Generative Models

---

## Fast Sampling

- https://arxiv.org/abs/2010.02502 — Song, Meng, Ermon (2020) — DDIM
- https://arxiv.org/abs/2206.00927 — Lu et al. (2022) — DPM-Solver
- https://arxiv.org/abs/2211.01095 — Lu et al. (2022) — DPM-Solver++
- https://arxiv.org/abs/2202.09778 — Liu et al. (2022) — PNDM
- https://arxiv.org/abs/2202.00512 — Salimans & Ho (2022) — Progressive Distillation
- https://arxiv.org/abs/2305.10652 — Song et al. (2023) — Consistency Models

---

## Conditional Generation and Guidance

- https://arxiv.org/abs/2207.12598 — Ho & Salimans (2021) — Classifier-Free Guidance
- https://arxiv.org/abs/2105.05233 — Dhariwal & Nichol (2021) — Diffusion Models Beat GANs
- https://arxiv.org/abs/2204.06125 — Ramesh et al. (2022) — DALL·E 2
- https://arxiv.org/abs/2112.10752 — Rombach et al. (2022) — Latent Diffusion Models / Stable Diffusion
- https://arxiv.org/abs/2302.05543 — Zhang et al. (2023) — ControlNet
- https://arxiv.org/abs/2302.08453 — Brooks et al. (2023) — InstructPix2Pix

---

## Latent Diffusion and Architecture

- https://arxiv.org/abs/2307.01952 — Peebles & Xie (2023) — DiT: Diffusion Transformers
- https://arxiv.org/abs/2310.00426 — Podell et al. (2023) — SDXL
- https://arxiv.org/abs/2403.03206 — Esser et al. (2024) — Stable Diffusion 3 (Flow Matching)
- https://arxiv.org/abs/2409.07565 — Black Forest Labs (2024) — FLUX.1
- https://arxiv.org/abs/2112.01073 — Esser et al. (2021) — VQGAN

---

## Video and World Models

- https://arxiv.org/abs/2402.17177 — Brooks et al. (2024) — Sora (OpenAI technical report)
- https://arxiv.org/abs/2311.15127 — Blattmann et al. (2023) — Stable Video Diffusion
- https://arxiv.org/abs/2209.14430 — Singer et al. (2022) — Make-A-Video
- https://arxiv.org/abs/2310.10647 — Ho et al. (2022) — Video Diffusion Models
- https://arxiv.org/abs/2403.19268 — Zhu et al. (2024) — World Models Survey

---

## Flow Matching (TLH-2 Preview)

- https://arxiv.org/abs/2210.02747 — Lipman et al. (2022) — Flow Matching for Generative Modeling
- https://arxiv.org/abs/2209.03003 — Liu et al. (2022) — Rectified Flow
- https://arxiv.org/abs/2302.00482 — Albergo & Vanden-Eijnden (2023) — Building Normalizing Flows with Stochastic Interpolants
- https://arxiv.org/abs/2209.01379 — Ma et al. (2024) — SiT: Scalable Interpolant Transformers

---

## Consistency Models and Distillation

- https://arxiv.org/abs/2303.01469 — Song et al. (2023) — Consistency Models
- https://arxiv.org/abs/2310.14189 — Song & Dhariwal (2023) — Improved Consistency Training
- https://arxiv.org/abs/2310.04378 — Kim et al. (2023) — CTM: Consistency Trajectory Models
- https://arxiv.org/abs/2311.17042 — Luo et al. (2023) — LCM: Latent Consistency Models

---

## Applications: Audio, Molecules, Biology (TLH-4 Preview)

- https://arxiv.org/abs/2009.09761 — Chen et al. (2020) — WaveGrad
- https://arxiv.org/abs/2009.09389 — Kong et al. (2020) — DiffWave
- https://arxiv.org/abs/2209.10049 — Watson et al. (2022) — RFdiffusion (protein backbone)
- https://arxiv.org/abs/2206.01729 — Hoogeboom et al. (2022) — Equivariant Diffusion for Molecule Generation (EDM)
- https://arxiv.org/abs/2209.15003 — Schneuing et al. (2022) — DiffSBDD

---

## Official Code Repositories

- https://github.com/hojonathanho/diffusion — Ho et al. DDPM (original TF)
- https://github.com/openai/improved-diffusion — Nichol & Dhariwal improved DDPM
- https://github.com/ermongroup/ddim — DDIM sampling
- https://github.com/openai/consistency_models — Song et al. Consistency Models
- https://github.com/CompVis/latent-diffusion — Rombach et al. LDM / Stable Diffusion
- https://github.com/huggingface/diffusers — HuggingFace Diffusers library
- https://github.com/lucidrains/denoising-diffusion-pytorch — Clean PyTorch DDPM (Phil Wang)
- https://github.com/dome272/Diffusion-Models-pytorch — Simple scratch implementation

---

## Tutorial Posts and Explainers

- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — Lilian Weng's canonical overview
- https://huggingface.co/blog/annotated-diffusion — Annotated PyTorch DDPM
- https://yang-song.net/blog/2021/score/ — Yang Song on score-based models
- https://nn.labml.ai/diffusion/ddpm/ — labml.ai annotated implementation
- https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869 — DDPM → DDIM → CFG bridge
- https://sander.ai/2022/01/08/score-based-generative-modeling.html — Sander Dieleman deep dive
- https://scorebasedgenerativemodeling.github.io — Song's interactive score-based modeling site
- https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/ — Introduction with visuals
- https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html — Cambridge MLG on flow matching
- https://diffusion.csail.mit.edu — MIT 6.S184 diffusion course materials

---

## Video Resources

- https://www.youtube.com/watch?v=HoKDTa5jHvg — Outlier: Diffusion Models Visual Intuition
- https://www.youtube.com/watch?v=fbLgFrlTnGU — Ari Seff: DDPM Explained
- https://www.youtube.com/watch?v=XCmphbFR1U8 — Yang Song: Score-Based Models (60 min)
- https://www.youtube.com/watch?v=1CIpzeNxIhU — Stable Diffusion Architecture Deep Dive
- https://www.youtube.com/watch?v=qqVCVWdXdbs — Consistency Models Explained
- https://www.youtube.com/watch?v=5ZSwYogAxYg — Yaron Lipman: Flow Matching
- https://www.youtube.com/watch?v=wMmqCMwuM2Q — CVPR 2022 Tutorial on Diffusion Models
- https://www.youtube.com/watch?v=pea3sH6orMc — Karras et al. EDM walk-through

---

## Benchmarks and Leaderboards

- https://paperswithcode.com/sota/image-generation-on-cifar-10 — CIFAR-10 FID leaderboard
- https://paperswithcode.com/sota/image-generation-on-imagenet-256x256 — ImageNet 256 FID
- https://paperswithcode.com/sota/text-to-image-generation-on-coco — Text-to-image COCO benchmarks
- https://huggingface.co/spaces/multimodalart/LoraTheExplorer — Model exploration tool
- https://stability.ai/research — Stability AI research papers

---

## HuggingFace Resources

- https://huggingface.co/docs/diffusers/index — Diffusers documentation
- https://huggingface.co/google/ddpm-cifar10-32 — Pre-trained DDPM CIFAR-10
- https://huggingface.co/google/ddpm-celebahq-256 — Pre-trained DDPM CelebA-HQ
- https://huggingface.co/stabilityai/stable-diffusion-2-1 — Stable Diffusion 2.1
- https://huggingface.co/black-forest-labs/FLUX.1-schnell — FLUX.1 flow matching model
- https://huggingface.co/docs/diffusers/tutorials/basic_training — Training your own DDPM

---

## Interactive Demos and Notebooks

- https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb — HF training tutorial Colab
- https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb — Google VDM demo
- https://github.com/cloneofsimo/minDiffusion — Minimal DDPM, ~200 lines
- https://github.com/explainingai-code/DDPM-Pytorch — Step-by-step DDPM tutorial

---

## Key Surveys

- https://arxiv.org/abs/2209.00796 — Croitoru et al. (2022) — Diffusion Models in Vision: A Survey
- https://arxiv.org/abs/2301.10972 — Yang et al. (2023) — Diffusion Models: A Comprehensive Survey
- https://arxiv.org/abs/2311.09608 — Cao et al. (2023) — Survey on Video Diffusion Models
