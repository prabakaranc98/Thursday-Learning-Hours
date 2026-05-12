import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """2-D spectrogram → sequence of patch embeddings."""

    def __init__(self, in_chans=1, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # (B, D, H', W')
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2)       # (B, H'*W', D)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """Small ViT operating on log-mel spectrogram patches. Returns CLS token."""

    def __init__(self, img_size=(128, 501), patch_size=16, embed_dim=256, depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        assert img_size[0] % patch_size == 0, "height must be divisible by patch_size"
        n_h = img_size[0] // patch_size
        n_w = math.ceil(img_size[1] / patch_size)
        n_patches = n_h * n_w

        self.patch_embed = PatchEmbed(1, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        tokens = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)              # (B, N+1, D)
        # Trim/pad positional embedding if spec width varies slightly
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)[:, 0]                        # CLS → (B, D)


class BaselineClassifier(nn.Module):
    """Step 1 baseline: AudioEncoder + linear multi-label head."""

    def __init__(self, img_size=(128, 501), patch_size=16, embed_dim=256,
                 depth=4, num_heads=4, dropout=0.1, num_classes=182):
        super().__init__()
        self.encoder = AudioEncoder(img_size, patch_size, embed_dim, depth, num_heads, dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))    # logits (B, num_classes)
