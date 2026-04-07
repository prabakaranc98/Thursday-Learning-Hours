# Resources — TLH-1: Applied Models: Diffusion Models from Scratch

> Curated tools, libraries, datasets, and courses for building and understanding diffusion models.

---

## Core Libraries

| Library | Purpose | Notes |
|---|---|---|
| **PyTorch** | All demos built with PyTorch | `pip install torch torchvision` |
| **HuggingFace Diffusers** | Production diffusion pipelines | `pip install diffusers` — reference for architecture |
| **HuggingFace Datasets** | MNIST, CIFAR-10, etc. | `pip install datasets` |
| **Accelerate** | Multi-GPU training | Not needed for MNIST demos |
| **einops** | Tensor rearrangement | Makes U-Net code readable |
| **tqdm** | Training progress bars | Included in requirements |
| **matplotlib** | Visualization + trajectory plots | Standard |
| **Pillow** | Image I/O | Standard |
| **wandb** | Experiment tracking | Optional — log loss curves |

---

## Reference Implementations

| Repo | Description |
|---|---|
| [openai/improved-diffusion](https://github.com/openai/improved-diffusion) | Official Nichol & Dhariwal implementation — production-quality U-Net |
| [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | Original Ho et al. DDPM — TensorFlow, but the canonical reference |
| [ermongroup/ddim](https://github.com/ermongroup/ddim) | Song et al. DDIM — deterministic sampling |
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | Phil Wang's clean PyTorch DDPM — best for learning |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | Production library — study the schedulers and UNet2DModel |
| [dome272/Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch) | Simple DDPM from scratch — closest to what we implement |
| [acids-ircam/flow_synthesizer](https://github.com/acids-ircam/flow_synthesizer) | Flow-based models reference — preview of TLH-2 |

---

## Datasets

| Dataset | Size | Use |
|---|---|---|
| **MNIST** | 70k, 28×28, 10 classes | Our demo — fits on CPU, trains in minutes |
| **FashionMNIST** | 70k, 28×28, 10 classes | Drop-in replacement for MNIST — slightly harder |
| **CIFAR-10** | 60k, 32×32, 10 classes | Next step up — needs GPU for good results |
| **CelebA** | 200k, 64×64, faces | Standard benchmark for image quality metrics |
| **ImageNet 64×64** | 1.2M | Standard for FID evaluations |

```python
# Download MNIST with torchvision
from torchvision import datasets, transforms
dataset = datasets.MNIST(root='./data', download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
    ]))
```

---

## Pre-trained Models for Exploration

| Model | Access | Purpose |
|---|---|---|
| **Stable Diffusion 1.5** | HuggingFace Hub | Reference implementation for LDM architecture |
| **DDPM CIFAR-10** | HuggingFace Hub | `google/ddpm-cifar10-32` |
| **DDPM CelebA-HQ** | HuggingFace Hub | `google/ddpm-celebahq-256` |
| **FLUX.1-schnell** | HuggingFace Hub | Flow matching model — TLH-2 preview |

```python
# Load a pre-trained DDPM and sample from it
from diffusers import DDPMPipeline
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
image = pipeline().images[0]
```

---

## Metrics

| Metric | Measures | Implementation |
|---|---|---|
| **FID** (Fréchet Inception Distance) | Sample quality vs real distribution | `pip install clean-fid` |
| **IS** (Inception Score) | Sample quality + diversity | Built into most eval suites |
| **CLIP Score** | Text-image alignment | HuggingFace metrics |
| **Precision/Recall** | Coverage vs quality trade-off | `prdc` package |

For MNIST demos: visual inspection + loss curves are sufficient.

---

## Key Blog Posts and Tutorials

| Resource | Author | Why Read |
|---|---|---|
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Lilian Weng | The canonical tutorial — math + diagrams, start here |
| [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) | HuggingFace | Line-by-line PyTorch implementation with explanation |
| [DDPM from Scratch](https://nn.labml.ai/diffusion/ddpm/) | labml.ai | Annotated PyTorch implementation |
| [Score-Based Generative Modeling](https://yang-song.net/blog/2021/score/) | Yang Song | The score-based SDE perspective, beautifully explained |
| [Diffusion Models Tutorial](https://jakiw.com/ddpm_derivation) | Jacob Wilson | Step-by-step ELBO → denoising objective derivation |
| [Understanding DDIM](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869) | — | Bridges DDPM → DDIM with worked examples |
| [Flow Matching for Generative Modeling](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) | Cambridge MLG | Preview of TLH-2 — how FM relates to diffusion |

---

## Video Lectures

| Resource | Duration | Content |
|---|---|---|
| [Diffusion Models (Outlier)](https://www.youtube.com/watch?v=HoKDTa5jHvg) | 30 min | Visual intuition — good starting point |
| [DDPM Lecture (Ari Seff)](https://www.youtube.com/watch?v=fbLgFrlTnGU) | 20 min | Mathematical derivation, very clear |
| [Score Matching (Song)](https://www.youtube.com/watch?v=XCmphbFR1U8) | 60 min | Yang Song's own lecture on score-based models |
| [Stable Diffusion Deep Dive](https://www.youtube.com/watch?v=1CIpzeNxIhU) | 45 min | Architecture walkthrough by Andrej Karpathy |
| [Consistency Models (Song)](https://www.youtube.com/watch?v=qqVCVWdXdbs) | 30 min | One-step generation explained |
| [Flow Matching (Lipman)](https://www.youtube.com/watch?v=5ZSwYogAxYg) | 50 min | Yaron Lipman's lecture on flow matching — TLH-2 prep |

---

## Evaluation Tools

| Tool | Purpose |
|---|---|
| `clean-fid` | FID computation against real data distributions |
| `torchmetrics` | IS, FID, LPIPS in PyTorch |
| `diffusers` benchmark scripts | Sampling speed vs quality measurements |

---

## Hardware Notes

| Hardware | MNIST Demo | CIFAR-10 | SD 1.5 |
|---|---|---|---|
| **CPU (laptop)** | ✓ ~5 min/10 epochs | ~4 hrs | Not feasible |
| **GPU (T4, free Colab)** | ~30 sec | ~20 min | ✓ feasible |
| **GPU (A100)** | ~5 sec | ~3 min | ~15 sec/image |

Our demos are designed to run on CPU. No GPU required.
