# Session Notes — TLH-3: Applied Models: Diffusion Transformers + Video & World Models

> **Date:** Apr 24, 2026 · **Format:** Foundations → Frontiers → Demo → Discussion

---

## 🧠 Core Thesis

> *"The U-Net was borrowed from biomedical image segmentation. It works — but it doesn't scale the way transformers do. DiT replaces the U-Net backbone with a Vision Transformer: patchify, attend globally, and condition on time and class via adaptive layer norm. The same scaling laws that made GPT-4 apply here. And extending that to video is conceptually simple: time is just another patch dimension."*

---

## 🏛️ FOUNDATIONS

### Section 1 — Why U-Net Falls Short at Scale

The convolutional U-Net:
- Has inductive biases (locality, translation equivariance) that help at small scale
- But self-attention only at coarse resolutions (16×16, 8×8) — limited global context
- Scaling requires doubling channels per level — quadratic compute growth
- Harder to adopt architecture improvements from LLM scaling

The Vision Transformer (ViT):
- Global attention at every layer — no scale bottleneck
- Follows power-law scaling with compute (same as language models)
- Easily extended to multi-modal inputs (text + image, image + video)
- DiT demonstrates this translates directly to diffusion

---

### Section 2 — DiT Architecture (Peebles & Xie, ICCV 2023)

**Patchification:**
```
Input: x_t ∈ ℝ^{H×W×C}  →  patches ∈ ℝ^{(H/p × W/p) × (p²C)}
```
Typical patch size $p = 2$ for 32×32 images → $16^2 = 256$ tokens.

**DiT Block:**
```
x → LayerNorm → Self-Attention → x
x → adaLN(t, c) → MLP → x
```

**adaLN (Adaptive Layer Norm):**

Instead of adding time embeddings via shift (like U-Net ResBlock), DiT learns per-token scale and shift from the time+class embedding:
$$\text{adaLN}(x, t, c) = \gamma(t, c) \cdot \text{LayerNorm}(x) + \delta(t, c)$$

where $\gamma, \delta$ are learned from a shared MLP over $[t \oplus c]$.

**Unpatchification:**
```
N tokens → Linear → p × p × C patches → rearrange → predicted noise ε̂
```

**DiT scales as a standard ViT:** DiT-S (33M), DiT-B (130M), DiT-L (458M), DiT-XL (675M).
FID improves monotonically with GFLOPs — follows a clean power law.

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, t_dim):
        super().__init__()
        self.norm1    = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn     = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2    = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp      = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        # adaLN — 6 outputs: scale/shift for pre-attn, post-attn, pre-mlp
        self.adaln    = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 6 * hidden_size))

    def forward(self, x, t_emb):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(t_emb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x
```

---

### Section 3 — DiT vs U-Net Comparison

| Property | U-Net | DiT |
|---|---|---|
| Attention scope | Only at coarse resolutions | Every layer, all tokens |
| Parameter scaling | Quadratic (channel doubling) | Linear (depth/width) |
| FID scaling | Diminishing returns | Power-law with GFLOPs |
| Skip connections | Essential (preserve detail) | Not needed (attention handles) |
| Inductive bias | Strong spatial locality | Minimal — general purpose |
| Adaptation to video | Hard — 3D convolutions needed | Easy — just add temporal tokens |

---

## 🔭 FRONTIERS

### Section 4 — Video Diffusion Models

**Naive extension:** Treat a video $V \in \mathbb{R}^{T \times H \times W \times C}$ as a tensor and apply standard 3D diffusion.

**Ho et al. (2022) — Video Diffusion Models:**
- Factorized attention: spatial attention over $(H, W)$ + temporal attention over $T$
- Joint training on images and videos — image generation is a special case ($T=1$)

**Blattmann et al. — Stable Video Diffusion:**
- Fine-tune image LDM with temporal layers inserted
- Temporal attention modules added at each resolution
- Motion conditioning via optical flow or CLIP video embeddings

**Architecture pattern:**
```
Video ∈ R^{T×H×W×C} → [VAE encode each frame] → z ∈ R^{T×h×w×4}
→ [Spatial attention over h×w at each t]
→ [Temporal attention over T at each spatial position]
→ [Decode] → output video
```

---

### Section 5 — Sora: Spacetime Patches

Brooks et al. (2024) — Technical report on OpenAI's video model.

**Key architectural insight:** Treat video as a sequence of **spacetime patches**:
```
Video V ∈ R^{T×H×W×C}  →  patch along all three dimensions
Spacetime patch: Δt × Δh × Δw  →  single token
```

This allows:
1. Training on videos of different resolutions and durations without architectural changes
2. Generating at any resolution/duration at test time
3. Learning both spatial and temporal structure in the same transformer

**Sora-class capabilities (inferred from technical report):**
- World-consistent long-form video (minutes)
- 3D consistency without explicit 3D representation
- Physics-like dynamics from scale alone

---

### Section 6 — Diffusion as World Models

A world model $W: (s_t, a_t) \to s_{t+1}$ predicts the next state given current state and action.

**DIAMOND (Alonso et al., 2024) — Diffusion for World Models:**

Uses a diffusion model to model $p(s_{t+1} | s_t, a_t)$:
- Frame $s_t$ is a latent image
- Action $a_t$ conditions the denoiser (like classifier-free guidance)
- Sample $s_{t+1}$ by running DDPM reverse with $a_t$ as context

**Advantages over autoregressive/deterministic world models:**
- Multimodal predictions (multiple possible futures)
- Better handling of stochastic environments
- Naturally generates high-fidelity observations

**Connection to RL:**
```
World model → plan in imagination → extract policy
Diffusion world model → differentiable rollouts → policy gradient through imagination
```

---

### Section 7 — SiT (Scalable Interpolant Transformers)

Ma et al. (2024) — Combines:
- **Stochastic interpolant** framework (flow matching paths, unified with diffusion)
- **DiT** backbone
- Achieves FID 2.06 on ImageNet 256×256 — competitive with DiT-XL/2 + flow matching improvements

Shows that the choice of probability path (diffusion vs FM) matters even with fixed architecture — flow matching paths give better FID at same model size.

---

## 🛠️ FRAMEWORK: DiT FROM SCRATCH

### Minimal DiT for MNIST

```
Input: x_t ∈ R^{1×28×28}, t ∈ {0,...,T}
Patch size p=4 → tokens: (28/4)² = 49 tokens, dim = 4²×1 = 16 → project to hidden_dim
+ sinusoidal position encoding
+ time + class (digit label) embedding via adaLN
N × DiT blocks (self-attention + adaLN + MLP)
Linear head → 49 × 16 → reshape to 1×28×28
```

```python
class DiTMNIST(nn.Module):
    def __init__(self, patch_size=4, hidden=256, depth=6, heads=4, t_dim=256, n_classes=10):
        super().__init__()
        self.p       = patch_size
        n_patches    = (28 // patch_size) ** 2
        patch_dim    = patch_size ** 2 * 1         # MNIST: 1 channel

        self.patch_embed = nn.Linear(patch_dim, hidden)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, hidden) * 0.02)

        # Time + class conditioning
        self.t_mlp   = nn.Sequential(nn.Linear(hidden, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.c_embed = nn.Embedding(n_classes + 1, t_dim)  # +1 for null class (CFG)

        self.blocks  = nn.ModuleList([DiTBlock(hidden, heads, t_dim) for _ in range(depth)])
        self.norm    = nn.LayerNorm(hidden)
        self.head    = nn.Linear(hidden, patch_dim)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.p
        x = x.reshape(B, C, H//p, p, W//p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H//p)*(W//p), C*p*p)

    def unpatchify(self, x, H=28, W=28):
        B, N, D = x.shape
        p = self.p
        x = x.reshape(B, H//p, W//p, 1, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, 1, H, W)

    def forward(self, x, t, c=None):
        tokens = self.patch_embed(self.patchify(x)) + self.pos_embed
        t_emb  = self.t_mlp(sinusoidal_embedding(t, 256))
        if c is not None:
            t_emb = t_emb + self.c_embed(c)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        return self.unpatchify(self.head(self.norm(tokens)))
```

---

## 💡 Key Takeaways

1. **DiT proves ViT > U-Net at scale.** FID follows a power law with compute — U-Net doesn't.

2. **Attention is all you need — in time too.** Video diffusion is U-Net/DiT + temporal attention. No special architecture needed.

3. **Spacetime patches unify image and video.** Sora's key insight: a single token can span both space and time.

4. **Diffusion world models are multimodal.** Unlike deterministic/autoregressive models, diffusion world models sample multiple possible futures.

5. **SiT shows: architecture + path choice both matter.** DiT + flow matching outperforms DiT + DDPM — you need both upgrades.

---

## ❓ Discussion Questions

- Why does self-attention at every layer (ViT) outperform self-attention only at coarse resolutions (U-Net)?
- What does it mean to treat video generation as "a diffusion model over spacetime patches"? What implicit assumptions does this make?
- How does action-conditioning in DIAMOND differ from text-conditioning via CFG in Stable Diffusion?
- What are the limitations of diffusion world models for real-time RL (hint: sampling speed)?
- How would you combine DiT + flow matching + consistency distillation for a real-time video generator?
