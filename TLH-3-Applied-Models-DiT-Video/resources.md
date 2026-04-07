# Resources — TLH-3: Applied Models: Diffusion Transformers + Video & World Models

---

## Core Libraries

| Library | Purpose |
|---|---|
| **PyTorch** | All demos |
| **einops** | Tensor rearrangement — essential for patchify/unpatchify |
| **timm** | Vision transformer implementations — reference for DiT blocks |
| **torchvision** | Dataset loading and transforms |
| **decord** | Fast video decoding — for video dataset loading |
| **imageio** | GIF/MP4 generation from frame sequences |

```bash
pip install torch torchvision einops timm tqdm matplotlib
pip install imageio[ffmpeg] decord   # for video work
```

---

## Reference Implementations

| Repo | Description |
|---|---|
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | Official DiT — adaLN blocks, ImageNet training |
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | Includes DiT variant |
| [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) | Open-source Sora reproduction — spacetime patches |
| [stability-ai/stable-video-diffusion-img2vid](https://github.com/Stability-AI/generative-models) | Official SVD codebase |
| [THUDM/CogVideo](https://github.com/THUDM/CogVideo) | CogVideoX — open video generation |
| [eloialonso/diamond](https://github.com/eloialonso/diamond) | Official DIAMOND world model code |
| [google-deepmind/genie](https://github.com/google-deepmind/genie) | Genie interactive world model reference |

---

## Pre-trained Models

| Model | Access | Notes |
|---|---|---|
| **DiT-XL/2** | `facebook/DiT-XL-2-256` on HF Hub | ImageNet 256 — use for exploration |
| **Stable Video Diffusion** | `stabilityai/stable-video-diffusion-img2vid` | SVD — image-to-video |
| **CogVideoX-5B** | `THUDM/CogVideoX-5b` | Open 5B video model |
| **Wan-2.1** | `Wan-AI/Wan2.1-T2V-14B` | State-of-the-art open video |

```python
# Load DiT-XL and sample from it
from diffusers import DiTPipeline
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
class_label = 207  # golden retriever
image = pipe(class_labels=[class_label]).images[0]
```

---

## Key Blog Posts and Tutorials

| Resource | Author | Why Read |
|---|---|---|
| [DiT Explained](https://johnowhitaker.dev/tils/2023-01-31-dit.html) | Jonathan Whitaker | Walk-through of DiT architecture |
| [How Sora Works](https://openai.com/research/video-generation-models-as-world-simulators) | OpenAI | The official technical report |
| [Building a ViT from Scratch](https://github.com/lucidrains/vit-pytorch) | lucidrains | ViT implementation — foundation for DiT |
| [DIAMOND World Model Blog](https://diamond-wm.github.io) | Alonso et al. | Interactive demos + paper walk-through |
| [Open-Sora Tutorial](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_v1.md) | HPC AI Tech | Implementation details for open Sora |

---

## Datasets

| Dataset | Size | Use |
|---|---|---|
| **MNIST** | 70k images | DiT demo — fits on CPU |
| **CIFAR-10** | 60k images | Larger DiT test — needs GPU for good FID |
| **ImageNet 256** | 1.2M | DiT benchmark — GPU cluster for training |
| **UCF-101** | 13k videos, 101 classes | Video classification/generation benchmark |
| **Minecraft Dataset** | Large | DIAMOND training data — screen recordings |

---

## Video Lectures

| Resource | Duration | Content |
|---|---|---|
| [DiT Talk (Peebles)](https://www.youtube.com/watch?v=sPsEPJiSZoY) | 25 min | DiT author explains the architecture |
| [Sora Behind the Scenes](https://openai.com/research/sora) | Blog | Interactive examples + paper |
| [DIAMOND NeurIPS Talk](https://www.youtube.com/watch?v=somelink) | 30 min | World model for RL |
| [Video Generation Survey](https://www.youtube.com/watch?v=u6qT23fAGjE) | 60 min | Comprehensive arc from VideoLDM to Sora |
