"""
Step 4 Clean FC-JEPA — Model Components

FCJEPAv4:
  E_c  context encoder (trainable ViT) — sees real soundscapes
  E_t  target encoder (EMA copy, stop-grad) — sees label-conditioned clean audio
  P1   SlotAttentionV4 with per-slot learned mu
  P2   ComposePredictor
  clf  SlotClassifier (max+mean pool over slots)
  cls_head  direct linear from CLS token (bypass signal)

Key change vs Step 3:
  - Per-slot mu: slot_mu shape (1, K_max, D) instead of (1, 1, D)
  - No Mode 1 (BYOL) — training loop handles the view gap externally
  - sig_reg: SIGReg moment-matching proxy (LeWorldModel arXiv:2603.19312)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SIGReg ─────────────────────────────────────────────────────────────────────

def sig_reg(Z: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularizer — moment-matching proxy.

    For a standard Gaussian every 1D projection has zero skewness and zero
    excess kurtosis (Cramér-Wold theorem). We penalize deviations from both.
    O(B × n_proj) — much cheaper than the full Epps-Pulley O(B² × n_proj).

    Z : (B, D)  unit-normalized embeddings
    Returns scalar loss (gradient flows back through Z).
    """
    if Z.shape[0] < 4:
        return Z.new_zeros(()).requires_grad_(Z.requires_grad)
    v = F.normalize(torch.randn(Z.shape[1], n_proj, device=Z.device, dtype=Z.dtype), dim=0)
    y = Z @ v                                                    # (B, n_proj)
    y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + 1e-8)
    kurt = ((y ** 4).mean(0) - 3.0) ** 2   # excess kurtosis → 0 for Gaussian
    skew = (y ** 3).mean(0) ** 2           # skewness → 0 for Gaussian
    return (kurt + skew).mean()


# ── Encoder components ─────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class ViTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n2   = nn.LayerNorm(d)
        h         = int(d * mlp_ratio)
        self.mlp  = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x    = x + y
        return x + self.mlp(self.n2(x))


class AudioEncoder(nn.Module):
    """ViT encoder for log-mel patches.

    forward(x)                 → CLS token (B, D)
    forward(x, return_tokens)  → (CLS (B, D), patch tokens (B, N, D))
    """

    def __init__(
        self, n_mels: int = 128, n_frames: int = 501, patch_size: int = 16,
        d: int = 256, n_heads: int = 4, depth: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        n_h = n_mels  // patch_size
        n_w = math.ceil(n_frames / patch_size)
        N   = n_h * n_w

        self.patch_embed = PatchEmbed(d, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed   = nn.Parameter(torch.zeros(1, N + 1, d))
        self.blocks      = nn.ModuleList(
            [ViTBlock(d, n_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        t   = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        t   = torch.cat([cls, t], dim=1)
        t   = t + self.pos_embed[:, : t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        if return_tokens:
            return t[:, 0], t[:, 1:]   # (B, D), (B, N, D)
        return t[:, 0]                  # (B, D)


# ── Slot Attention V4 ──────────────────────────────────────────────────────────

class SlotAttentionV4(nn.Module):
    """
    Slot Attention (Locatello et al. 2020) with per-slot learned initialization.

    Step 3 used a single shared slot_mu (1, 1, D) broadcast to all K slots.
    Here slot_mu is (1, K_max, D) — each slot has a distinct learned origin,
    reducing symmetry-breaking pressure on the attention competition.

    Accepts an optional n_slots argument to activate fewer than K_max slots.
    Unused slots are simply not allocated, so their gradients are zero.
    """

    def __init__(self, K_max: int = 6, D: int = 256, n_iter: int = 3, eps: float = 1e-8):
        super().__init__()
        self.K_max  = K_max
        self.D      = D
        self.n_iter = n_iter
        self.eps    = eps
        self.scale  = D ** -0.5

        self.slot_mu        = nn.Parameter(torch.zeros(1, K_max, D))
        self.slot_log_sigma = nn.Parameter(torch.full((1, K_max, D), -2.0))

        self.norm_in    = nn.LayerNorm(D)
        self.norm_slots = nn.LayerNorm(D)
        self.norm_mlp   = nn.LayerNorm(D)

        self.to_k = nn.Linear(D, D, bias=False)
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.gru  = nn.GRUCell(D, D)
        self.mlp  = nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))

        nn.init.trunc_normal_(self.slot_mu, std=0.02)

    def forward(self, inputs: torch.Tensor, n_slots: int | None = None):
        """
        inputs  : (B, N, D)
        n_slots : number of active slots (≤ K_max); defaults to K_max
        Returns : slots (B, K, D), activations (B, K)
        """
        B = inputs.shape[0]
        K = min(n_slots, self.K_max) if n_slots is not None else self.K_max

        normed = self.norm_in(inputs)
        k = self.to_k(normed)   # (B, N, D)
        v = self.to_v(normed)   # (B, N, D)

        sigma = self.slot_log_sigma[:, :K].exp().expand(B, K, self.D)
        slots = self.slot_mu[:, :K].expand(B, K, self.D) + sigma * torch.randn_like(sigma)

        for _ in range(self.n_iter):
            prev   = slots
            q      = self.to_q(self.norm_slots(slots))                    # (B, K, D)
            dots   = torch.einsum("bkd,bnd->bkn", q, k) * self.scale     # (B, K, N)
            attn   = dots.softmax(dim=1)                                  # compete over slots
            attn_n = attn / (attn.sum(-1, keepdim=True) + self.eps)
            upd    = torch.einsum("bkn,bnd->bkd", attn_n, v)
            slots  = self.gru(
                upd.reshape(B * K, self.D),
                prev.reshape(B * K, self.D),
            ).reshape(B, K, self.D)
            slots  = slots + self.mlp(self.norm_mlp(slots))

        activations = attn.sum(-1)   # (B, K) — total attention mass per slot
        return slots, activations


# ── Composition predictor & classifier ────────────────────────────────────────

class ComposePredictor(nn.Module):
    """Weighted combination of slots → predicted composition embedding."""

    def __init__(self, D: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D),
        )

    def forward(self, slots: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        w = acts.softmax(-1).unsqueeze(-1)         # (B, K, 1)
        return self.mlp((slots * w).sum(1))        # (B, D)


class SlotClassifier(nn.Module):
    """Per-slot classification head with max+mean pooling."""

    def __init__(self, D: int = 256, n_cls: int = 234):
        super().__init__()
        self.head = nn.Linear(D, n_cls)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        per_slot = self.head(slots)                # (B, K, n_cls)
        return (per_slot.max(dim=1).values + per_slot.mean(dim=1)) * 0.5


# ── Full model ─────────────────────────────────────────────────────────────────

class FCJEPAv4(nn.Module):
    """
    Clean FC-JEPA v4.

    During training the caller passes spec_ctx to encode_ctx and
    spec_target_* to encode_target (with stop-grad). The training loop
    computes all losses externally using the returned embeddings.

    forward() is the inference path: spec_ctx → logits.
    """

    def __init__(
        self,
        n_mels: int, n_frames: int, patch_size: int, D: int,
        n_heads: int, depth: int, dropout: float,
        k_max: int, n_slot_iter: int, n_cls: int,
    ):
        super().__init__()
        enc_args = (n_mels, n_frames, patch_size, D, n_heads, depth, dropout)
        self.enc_ctx = AudioEncoder(*enc_args)
        self.enc_tgt = copy.deepcopy(self.enc_ctx)
        for p in self.enc_tgt.parameters():
            p.requires_grad_(False)

        self.P1      = SlotAttentionV4(k_max, D, n_slot_iter)
        self.P2      = ComposePredictor(D)
        self.clf     = SlotClassifier(D, n_cls)
        self.cls_head = nn.Linear(D, n_cls)

    @torch.no_grad()
    def ema_update(self, tau: float = 0.996) -> None:
        for p_c, p_t in zip(self.enc_ctx.parameters(), self.enc_tgt.parameters()):
            p_t.data.mul_(tau).add_((1.0 - tau) * p_c.data)

    def encode_ctx(self, x: torch.Tensor):
        """Returns (cls, tokens) — both require_grad."""
        return self.enc_ctx(x, return_tokens=True)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-normed EMA target embedding, stop-grad."""
        return F.normalize(self.enc_tgt(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: (B, 1, H, W) → (B, n_cls) logits."""
        cls, tokens = self.enc_ctx(x, return_tokens=True)
        slots, acts = self.P1(tokens)
        return self.cls_head(cls) + self.clf(slots)
