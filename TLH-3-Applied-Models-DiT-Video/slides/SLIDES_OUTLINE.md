# Slides Outline — TLH-3: Diffusion Transformers + Video & World Models

> **19 slides · 60-minute session**
> Foundations (15 min) → Frontiers (15 min) → Demo (20 min) → Discussion (10 min)

---

## Slide 1 — Cover
**Title:** Diffusion Transformers + Video & World Models
**Subtitle:** From U-Net to ViT — scaling diffusion to video and environment simulation
**Session:** TLH-3 · Thursday Learning Hours · Applied Models Track · Part 3 of 3
*Visual: Grid showing progression — MNIST digit → ImageNet class → video frame sequence → 3D world*

---

## 🏛️ FOUNDATIONS (15 min, Slides 2–6)

## Slide 2 — The U-Net's Hidden Limitation
**Heading:** Why the U-Net Doesn't Scale Like Language Models

U-Net properties:
- Local convolutions + skip connections → preserves fine detail
- Self-attention only at coarse resolutions (8×8, 16×16)
- Scaling: double channels per level → quadratic cost

ViT (Vision Transformer) properties:
- Global attention from layer 1 across all patches
- Predictable scaling: FID ∝ GFLOPs^{-α} — a power law
- Same scaling laws as GPT-4, Gemini, LLaMA

*"U-Net is a hand-crafted inductive bias. ViT is a learnable inductive bias. At scale, learnable wins."*

---

## Slide 3 — DiT: The Architecture
**Heading:** Patchify → ViT Blocks → Unpatchify

```
x_t ∈ ℝ^{H×W×C}
    ↓ patchify (p×p patches → linear → hidden tokens)
    ↓ + sinusoidal position embedding
[DiT Block × L]
    ↓ adaLN(t, c) → Self-Attention → adaLN → MLP
    ↓ LayerNorm → Linear head
    ↓ unpatchify
ε̂ ∈ ℝ^{H×W×C}    (predicted noise)
```

**adaLN — Adaptive Layer Norm:**
$$\text{adaLN}(x, t, c) = \gamma(t,c) \cdot \text{LN}(x) + \delta(t,c)$$

$\gamma, \delta$ come from a shared MLP over the time+class embedding — learned per-token scale and shift.

*Visual: One DiT block diagram with labeled components*

---

## Slide 4 — DiT Scaling Laws
**Heading:** FID follows a Power Law with GFLOPs

*Visual: Figure from Peebles & Xie 2023 — FID vs GFLOPs for DiT-S, B, L, XL*

| Model | Params | GFLOPs | FID (ImageNet 256) |
|---|---|---|---|
| DiT-S/2 | 33M | 6 | 68.4 |
| DiT-B/2 | 130M | 23 | 43.5 |
| DiT-L/2 | 458M | 80 | 23.3 |
| DiT-XL/2 | 675M | 118 | 2.27 |

Compare: U-Net counterparts plateau — adding parameters stops helping.

*"DiT doesn't have a ceiling. U-Net does."*

---

## Slide 5 — DiT vs U-Net: The Design Decision
**Heading:** What Each Architecture Gets Right

| Property | U-Net | DiT |
|---|---|---|
| Skip connections | Essential (preserve spatial detail) | Not needed — attention handles it |
| Attention scope | Coarse resolutions only | Every layer, all tokens |
| Time conditioning | Shift in ResBlock | adaLN in every DiT block |
| Video extension | Requires 3D convolutions | Add temporal tokens naturally |
| Scaling | Diminishing returns | Power law |
| Architecture complexity | Many hand-crafted choices | One principled design |

---

## Slide 6 — From 2D to Video: Spacetime Patches
**Heading:** Time is Just Another Dimension

Image: $x \in \mathbb{R}^{H \times W \times C}$ → patches of size $p \times p$

Video: $x \in \mathbb{R}^{T \times H \times W \times C}$ → spacetime patches of size $\Delta t \times p \times p$

Same DiT block — now attends to patches across both space and time. The model doesn't "know" which dimension is time.

**Key insight from Sora:**
- Variable $T$, $H$, $W$ — just change the number of tokens
- Train on diverse resolutions/durations in the same batch
- No architectural modification for different aspect ratios

*Visual: 2D patch grid vs 3D spacetime patch grid*

---

## 🔭 FRONTIERS (15 min, Slides 7–11)

## Slide 7 — Sora: What We Know
**Heading:** Video Generation Models as World Simulators (OpenAI, 2024)

Key claims from the technical report:
1. **Spacetime patches** — videos are sequences of $\Delta t \times p \times p$ tokens
2. **Variable-length training** — same model generates 1s clips and 1min videos
3. **Emergent 3D consistency** — objects maintain geometry without explicit 3D supervision
4. **Physics** — liquids, cloth, reflections emerge from video prediction

Architecture (inferred): latent video DiT + flow matching (not confirmed — likely based on SD3/OpenAI history)

*Visual: Sora sample frames — diverse settings, long temporal coherence, physics*

---

## Slide 8 — Stable Video Diffusion
**Heading:** Fine-Tuning Image Models for Video

Starting point: image LDM (Stable Diffusion)
Modification: insert temporal attention layers at each resolution

```
For each frame t in video:
    z_t = spatial_attention(z_t)       # original image layers
    z_t = temporal_attention([z_{1:T}])  # new: attend across frames at same spatial position
```

Training:
1. Phase 1: image pretraining
2. Phase 2: video fine-tuning on curated video dataset
3. Phase 3: high-res + quality fine-tuning

*This is the production recipe for most open video models (CogVideoX, Wan, Open-Sora).*

---

## Slide 9 — DIAMOND: Diffusion World Model for RL
**Heading:** Generate the Next Frame — Action-Conditioned

**World model role:** $p(s_{t+1} | s_t, a_t)$ — predict next observation from current state + action.

**DIAMOND's approach:**
- Use DDPM to model $p(s_{t+1} | s_t, a_t)$
- Action $a_t$ conditions the denoiser via cross-attention (like text conditioning in Stable Diffusion)
- State history $[s_{t-k}, ..., s_t]$ as context tokens

**Results on Atari (DIAMOND, NeurIPS 2024):**
- Better visual quality than autoregressive world models
- Comparable policy performance to DreamerV3
- Handles stochastic environments naturally — samples multiple possible futures

*Visual: DIAMOND-generated Atari frames vs real frames — nearly indistinguishable*

---

## Slide 10 — Genie: Interactive World Generation
**Heading:** Generate a Playable World from a Single Image

Bruce et al. (DeepMind, 2024):
- Given a single image (or text prompt), generate an interactive environment
- 11B parameter spatiotemporal transformer
- Learns latent action space from unlabeled video — no action labels needed!
- Actions are discovered, not specified

**Architecture:**
1. Video tokenizer: encode frames as discrete tokens
2. Latent action model: discover actions from video transitions
3. Dynamics model: predict next frame given current frame + latent action

*Visual: Genie generates multiple interactable environments from web scraped videos*

---

## Slide 11 — The Landscape: DiT × Video × World Models
**Heading:** Where the Field Is Going

| Model | Architecture | Training | Use Case |
|---|---|---|---|
| **DiT** | ViT + adaLN | DDPM + FM | Image generation, scaling |
| **Stable Video Diffusion** | U-Net + temporal attn | Video FT | Short video (4s) |
| **Sora** | Video DiT + FM | Large-scale | Long video (1min+) |
| **DIAMOND** | U-Net + cross-attn | DDPM on Atari | RL world model |
| **Genie** | Spatiotemporal ViT | Self-supervised | Interactive environment |
| **Movie Gen** | DiT (Meta) | FM on video | Video + audio + image |

---

## 🛠️ DEMO (20 min, Slides 12–16)

## Slide 12 — What We Build
**Heading:** Demo Goal: Minimal DiT on MNIST + Temporal Extension

```
Demo 01 — DiT on MNIST:
  ✓ Patchify: 28×28 → 49 tokens of dim 16 → project to hidden_dim=256
  ✓ Sinusoidal position embedding
  ✓ Time + class embedding via adaLN
  ✓ 6 × DiT blocks (self-attention + adaLN + MLP)
  ✓ Class-conditional generation (digit 0–9)

Demo 02 — Temporal extension:
  ✓ Extend patchify to time dimension
  ✓ Generate sequences of consecutive digits
  ✓ Temporal self-attention over frames
```

---

## Slide 13 — Live Demo: Part 1 — DiT Block
**Heading:** The adaLN Block in Code

```python
class DiTBlock(nn.Module):
    def __init__(self, dim, heads, t_dim):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn   = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2  = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp    = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        # 6 outputs: (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaln  = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 6*dim))

    def forward(self, x, t_emb):
        s1,sc1,g1, s2,sc2,g2 = self.adaln(t_emb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1[:, None]) + s1[:, None]
        h, _ = self.attn(h, h, h)
        x = x + g1[:, None] * h
        h = self.norm2(x) * (1 + sc2[:, None]) + s2[:, None]
        x = x + g2[:, None] * self.mlp(h)
        return x
```

*Show: attention maps — global from layer 1 (unlike U-Net which is local until coarse levels)*

---

## Slide 14 — Live Demo: Part 2 — Class-Conditional Generation
**Heading:** Digit-Conditioned Sampling with CFG

```python
# Class-conditional DiT with classifier-free guidance
model = DiTMNIST(n_classes=10)

# Training: randomly drop class label (CFG)
c = label if random() > 0.1 else torch.full_like(label, 10)  # 10 = null class

# Sampling with CFG at w=3.0
eps_cond   = model(x_t, t, c=label)
eps_uncond = model(x_t, t, c=null_class)
eps_guided = (1 + w) * eps_cond - w * eps_uncond
```

*Show: 10×8 grid of generated digits — 8 samples per class, clear class separation*

---

## Slide 15 — Live Demo: Part 3 — Temporal Extension
**Heading:** Adding Time as a Patch Dimension

```python
# Temporal patchify: treat [frame0, frame1, ..., frameT] as extra patch dim
def temporal_patchify(video):  # video: (B, T, C, H, W)
    B, T, C, H, W = video.shape
    patches = []
    for t_idx in range(T):
        patches.append(patchify(video[:, t_idx]))   # spatial patches per frame
    # stack: (B, T * n_spatial_patches, patch_dim)
    return torch.cat(patches, dim=1)

# Add temporal positional encoding
temporal_pos = torch.arange(T).repeat_interleave(n_patches)
```

*Show: generated digit sequences — consecutive frames with temporal coherence*

---

## Slide 16 — Scaling Analysis: U-Net vs DiT on MNIST
**Heading:** Even on MNIST — DiT Scales More Predictably

*Visual: 2×2 grid*
- Loss curves: U-Net (levels out early) vs DiT (continues to improve with depth)
- FID vs parameter count: U-Net plateau vs DiT linear improvement
- Attention maps: U-Net (local at fine scales) vs DiT (global from layer 1)
- Generated samples quality vs model size: DiT improves more consistently

---

## 🔄 DISCUSSION (10 min, Slides 17–19)

## Slide 17 — The Architecture Arc
**Heading:** U-Net → DiT → Video → World Models

```
2020  U-Net           DDPM — practical but doesn't scale
2022  U-Net++         Improved skip connections, attention at all scales
2023  DiT             ViT backbone, adaLN, power-law scaling
2023  Video DiT       Add temporal attention dimension
2024  Sora            Spacetime patches, variable resolution, 1-min video
2024  World Models    Action-conditioned video DiT → RL agent brain
```

---

## Slide 18 — Open Questions

1. Why does global attention from layer 1 outperform hierarchical attention? Is it the expressivity or the optimization landscape?
2. Sora achieves 3D consistency without explicit 3D supervision. What is the implicit representation?
3. DIAMOND shows diffusion world models work — but they're slow (sampling takes time). How do you make a real-time world model?
4. Genie discovers action spaces from unlabeled video. What defines a "good" latent action space?
5. As video models approach photo-realism, how do you distinguish generated from real? Does it matter?

---

## Slide 19 — Takeaways + TLH-4 Preview

**Today:**
- DiT = patchify → ViT with adaLN → unpatchify — scales with power law
- Video = add temporal patches — same transformer, more tokens
- Sora = spacetime patches + variable resolution + scale → emergent physics
- World models = action-conditioned video diffusion → RL agent imagination

**TLH-4 — Diffusion for Biology:**
- The exact same framework — but the data is protein backbone frames (SO(3) rotations + ℝ³ positions)
- Forward process: IGSO(3) noise instead of Gaussian noise
- Denoiser: SE(3)-equivariant GNN instead of U-Net/DiT
- Output: protein backbones, small molecules, conformational ensembles
- RFdiffusion = AlphaFold2 + DDPM

---

## Design Notes
- **Colors:** ivory `#f7f3ea`, copper `#8f3e22`, charcoal text
- **Architecture diagrams:** clean box-and-arrow style, labeled dimensions
- **Attention maps:** visualise with heatmaps at each layer
