# Resources — TLH-2: Applied Models: Flow Matching

---

## Core Libraries

| Library | Purpose |
|---|---|
| **PyTorch** | All demos — same setup as TLH-1 |
| **torchdiffeq** | ODE solvers (RK45, Dopri5) — for adaptive-step FM sampling |
| **torchdyn** | Neural ODE library — higher-level FM/CNF tools |
| **pot (Python Optimal Transport)** | Minibatch OT coupling for OT-CFM |
| **zuko** | Flow matching library — production-quality CFM implementations |
| **HuggingFace Diffusers** | Stable Diffusion 3 / FLUX pipelines |

```bash
pip install torch torchvision tqdm matplotlib einops
pip install torchdiffeq pot zuko   # flow matching extras
pip install diffusers               # for SD3/FLUX exploration
```

---

## Reference Implementations

| Repo | Description |
|---|---|
| [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) | Official CFM library — OT-FM, Riemannian FM, all variants |
| [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | Phil Wang's clean rectified flow implementation |
| [gle-bellier/flow-matching](https://github.com/gle-bellier/flow-matching) | Minimal RF from scratch — excellent teaching reference |
| [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion) | Production-quality diffusion/FM codebase with schedulers |
| [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Official FLUX.1 code — flow matching in production |
| [Stability-AI/sd3-ref](https://github.com/Stability-AI/sd3-ref) | SD3 reference implementation |
| [openai/consistency_models](https://github.com/openai/consistency_models) | Consistency + reflow — 1-step generation |

---

## Key Blog Posts and Tutorials

| Resource | Author | Why Read |
|---|---|---|
| [Flow Matching Guide](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) | Cambridge MLG | Clear intro — RF vs CFM vs stochastic interpolants |
| [Rectified Flow Explained](https://www.cs.utexas.edu/~lqiang/rectflow.html) | Qiang Liu's group | Original RF authors' explainer |
| [Flow Matching Tutorial](https://neurips.cc/virtual/2023/tutorial/73945) | Lipman, Liu, Le | NeurIPS 2023 tutorial — 3 hours, authoritative |
| [Score to Flow: A Visual Guide](https://diffusion.csail.mit.edu) | MIT 6.S184 | Visual comparisons of score-based vs flow-based |
| [Understanding FLUX](https://blackforestlabs.ai/flux-1/) | Black Forest Labs | Architecture + training details |

---

## Pre-trained Models

| Model | Access | Notes |
|---|---|---|
| **FLUX.1-schnell** | `black-forest-labs/FLUX.1-schnell` on HF Hub | 4-step rectified flow, Apache 2.0 |
| **FLUX.1-dev** | `black-forest-labs/FLUX.1-dev` on HF Hub | Non-commercial, full quality |
| **Stable Diffusion 3 Medium** | `stabilityai/stable-diffusion-3-medium` on HF Hub | Open weights MM-DiT + RF |

```python
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
image = pipe("a photo of a cat", num_inference_steps=4).images[0]  # 4 steps!
```

---

## Metrics

| Metric | Use |
|---|---|
| FID | Sample quality vs real distribution |
| NFE (Number of Function Evaluations) | Sampling efficiency — lower = better |
| Path straightness | $\mathrm{straightness} = 1 - \frac{1}{T}\sum_t \|v_t - \bar v\|$ — 1.0 = perfectly straight |

---

## Video Lectures

| Resource | Duration | Content |
|---|---|---|
| [Yaron Lipman — Flow Matching](https://www.youtube.com/watch?v=5ZSwYogAxYg) | 50 min | CFM paper author's lecture |
| [Xingchao Liu — Rectified Flow](https://www.youtube.com/watch?v=nPqP8JNp9JI) | 40 min | RF author's explanation |
| [NeurIPS 2023 Tutorial on FM](https://neurips.cc/virtual/2023/tutorial/73945) | 3 hrs | Comprehensive coverage |
| [Flow Matching vs Diffusion](https://www.youtube.com/watch?v=DDq_pIfHqLs) | 25 min | Visual comparison |
