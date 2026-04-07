# Reference Links — TLH-2: Applied Models: Flow Matching

---

## Foundational Papers

- https://arxiv.org/abs/2210.02747 — Lipman et al. (2022) — Flow Matching for Generative Modeling
- https://arxiv.org/abs/2209.03003 — Liu, Gong, Liu (2022) — Rectified Flow (Flow Straight and Fast)
- https://arxiv.org/abs/2209.15571 — Albergo & Vanden-Eijnden (2022) — Stochastic Interpolants
- https://arxiv.org/abs/2302.00482 — Albergo, Boffi, Vanden-Eijnden (2023) — Stochastic Interpolants: A Unifying Framework
- https://arxiv.org/abs/2307.08698 — Pooladian et al. (2023) — Multisample Flow Matching

## Optimal Transport and CFM

- https://arxiv.org/abs/2302.05872 — Tong et al. (2023) — Improving and Generalizing Flow-Matching Models (OT-CFM)
- https://arxiv.org/abs/2310.02894 — Chen & Lipman (2023) — Flow Matching on General Geometries (Riemannian FM)
- https://arxiv.org/abs/2407.15595 — Campbell et al. (2024) — Discrete Flow Matching
- https://pot.readthedocs.io — Python Optimal Transport library documentation

## Production Models

- https://arxiv.org/abs/2403.03206 — Esser et al. (2024) — Stable Diffusion 3 (MM-DiT + RF)
- https://arxiv.org/abs/2409.07565 — Black Forest Labs (2024) — FLUX.1
- https://arxiv.org/abs/2401.08740 — Ma et al. (2024) — SiT: Scalable Interpolant Transformers
- https://blackforestlabs.ai/flux-1/ — FLUX.1 blog post
- https://stability.ai/news/stable-diffusion-3-research-paper — SD3 blog

## Distillation and Fast Sampling

- https://arxiv.org/abs/2303.01469 — Song et al. (2023) — Consistency Models
- https://arxiv.org/abs/2410.12557 — Shortcut Models (2024)
- https://arxiv.org/abs/2311.17042 — Luo et al. (2023) — LCM: Latent Consistency Models
- https://arxiv.org/abs/2311.13608 — Sauer et al. (2023) — SDXL-Turbo / ADD

## Applications Beyond Images

- https://arxiv.org/abs/2306.15687 — Le et al. (Meta, 2023) — Voicebox
- https://arxiv.org/abs/2309.12230 — Vyas et al. (2023) — Audiobox
- https://arxiv.org/abs/2310.02391 — Bose et al. (2023) — FoldFlow
- https://arxiv.org/abs/2312.04557 — Bose et al. (2024) — SE(3)-SFM for Protein Backbone Generation
- https://arxiv.org/abs/2302.08929 — Chen & Lipman (2023) — Riemannian FM

## Official Code

- https://github.com/atong01/conditional-flow-matching — CFM library (the one to use)
- https://github.com/lucidrains/rectified-flow-pytorch — Clean RF implementation
- https://github.com/black-forest-labs/flux — FLUX.1 official code
- https://github.com/Stability-AI/generative-models — SD3 / SDXL codebase
- https://github.com/openai/consistency_models — Consistency + reflow

## Tutorials and Explainers

- https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html — Cambridge MLG FM tutorial
- https://www.cs.utexas.edu/~lqiang/rectflow.html — Rectified flow overview (original authors)
- https://neurips.cc/virtual/2023/tutorial/73945 — NeurIPS 2023 FM tutorial
- https://diffusion.csail.mit.edu — MIT 6.S184 diffusion + FM materials
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — Lilian Weng (background, DDPM → FM)

## Video

- https://www.youtube.com/watch?v=5ZSwYogAxYg — Yaron Lipman: Flow Matching (50 min)
- https://www.youtube.com/watch?v=nPqP8JNp9JI — Xingchao Liu: Rectified Flow (40 min)
- https://www.youtube.com/watch?v=DDq_pIfHqLs — FM vs Diffusion visual comparison

## HuggingFace Resources

- https://huggingface.co/black-forest-labs/FLUX.1-schnell — FLUX.1-schnell (Apache 2.0)
- https://huggingface.co/black-forest-labs/FLUX.1-dev — FLUX.1-dev (non-commercial)
- https://huggingface.co/stabilityai/stable-diffusion-3-medium — SD3 Medium
- https://huggingface.co/docs/diffusers/using-diffusers/sd3 — SD3 diffusers tutorial
