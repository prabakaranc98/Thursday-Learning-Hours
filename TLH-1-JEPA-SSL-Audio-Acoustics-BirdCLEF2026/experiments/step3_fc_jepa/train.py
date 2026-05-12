"""
Step 3: Factorize-and-Compose Audio-JEPA on BirdCLEF 2026.

Architecture
────────────────────────────────────────────────────────────
  E_c  context encoder (trainable ViT): audio → patch tokens + CLS
  E_t  target encoder (EMA copy, stop-grad): audio → unit-normed CLS
  P1   slot attention:  patch tokens → K slots + per-slot activations
  P2   compose predictor: slots + acts → predicted composition embedding
  Clf  slot classifier: per-slot Linear, max pooled → (B, n_cls) logits

Training modes
────────────────────────────────────────────────────────────
  Mode 1 – single clip or soundscape window (always active)
    x → E_c → (tokens, cls)
    x → E_t → z (unit-normed, stop-grad)
    slots, acts = P1(tokens)
    L_cls  = BCE(Clf(slots), labels)
    L_fact = MSE(argmin_k slot_k, z)          ← vectorized, no scipy
    L_div  = mean off-diagonal cosine similarity in slot Gram
    L_var  = VICReg variance on cls across batch

  Mode 2 – synthetic 3-component mix (optional, every mix_every steps)
    x_A, x_B, x_bg → E_t → z_A, z_B, z_bg   (stop-grad, unit-normed)
    z_comp = normalize(z_A + z_B + z_bg)
    x_mix = w_A*x_A + w_B*x_B + w_bg*x_bg    (spectrogram-space mix)
    x_mix → E_c → P1 → slots, acts
    L_fact = Hungarian-MSE(slots, {z_A, z_B, z_bg})
    L_comp = MSE(P2(slots, acts), z_comp)
    L_cls  = BCE(Clf(slots), labels_A ∪ labels_B)

Combined loss
    L = L_cls + λ_fact*L_fact + λ_comp*L_comp + λ_div*L_div + λ_var*L_var

EMA update (per step)
    θ_t ← τ·θ_t + (1-τ)·θ_c     τ = 0.996

Warm start: loads E_c weights from a Step 1 checkpoint (--step1_ckpt).
"""

import ast
import copy
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ── WandB (optional) ──────────────────────────────────────────────────────────
try:
    import wandb
    _wkey = None
    try:
        from kaggle_secrets import UserSecretsClient
        _wkey = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        _wkey = os.environ.get("WANDB_API_KEY")
    if _wkey:
        wandb.login(key=_wkey, relogin=True)
        _USE_WANDB = True
    else:
        _USE_WANDB = False
        print("[WANDB] WANDB_API_KEY not set — stdout only")
except Exception as _we:
    _USE_WANDB = False
    print(f"[WANDB] unavailable ({_we}) — stdout only")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Audio frontend
    sample_rate      = 32_000,
    n_mels           = 128,
    n_fft            = 1024,
    hop_length       = 320,
    duration         = 5.0,
    # ViT encoder
    d_model          = 256,
    n_heads          = 4,
    n_layers         = 4,
    patch_size       = 16,
    dropout          = 0.1,
    # Slot attention (P1)
    n_slots          = 8,
    n_slot_iter      = 3,
    # EMA target encoder
    ema_tau          = 0.996,
    # Training
    epochs           = 30,
    batch_size       = 64,       # overridden to 256 on A100
    lr               = 3e-4,
    weight_decay     = 0.05,
    warmup_epochs    = 3,
    log_every        = 1,
    # Loss weights
    # ── Loss weights ──────────────────────────────────────────────────────────
    # Primary JEPA objective: L_comp (composition MSE) + L_fact (Hungarian match)
    # These force P1 to factorize the mix into individual species representations.
    # L_cls is the auxiliary supervised signal (keeps enc_ctx relevant to birds).
    lambda_fact      = 0.5,
    lambda_comp      = 1.0,      # primary JEPA objective
    lambda_div       = 0.1,      # slot diversity (off-diagonal cosine) — slot anti-collapse
    lambda_var       = 0.1,      # VICReg variance on CLS — encoder anti-collapse
    encoder_lr_scale = 0.1,      # enc_ctx gets 10× lower LR than new heads
    # ── Mode 2 frequency ──────────────────────────────────────────────────────
    # Mode 2 is the TRUE JEPA signal (different views: mix vs individual clips).
    # Mode 1 L_fact (same clip → same clip EMA) is auxiliary/BYOL-like.
    # mix_every=2 → 50% of steps are Mode 2 (up from 33%).

    # Mode 2 (synthetic mix)
    mix_every        = 2,        # 0 = disable; 2 = 50% Mode 2 (primary JEPA)
    mix_batch_size   = 64,       # overridden to 128 on A100
    # Data
    data_root        = "/kaggle/input/birdclef-2026",
    num_workers      = 4,
    val_fraction     = 0.15,
    seed             = 42,
    secondary_weight = 0.5,
    min_rating       = 0.0,
    # Warm start
    step1_ckpt       = None,     # path to Step 1 best.pt
)

# ── DATA HELPERS (shared with step1) ─────────────────────────────────────────

def find_data_root(cfg_root):
    if os.path.isdir(cfg_root):
        return cfg_root
    base = "/kaggle/input"
    print(f"[DATA] '{cfg_root}' not found — scanning {base}")
    if not os.path.isdir(base):
        return cfg_root
    entries = sorted(os.listdir(base))
    print(f"[DATA] /kaggle/input/: {entries}")
    nested_base = os.path.join(base, "competitions")
    if os.path.isdir(nested_base):
        for d in sorted(os.listdir(nested_base)):
            if "bird" in d.lower():
                p = os.path.join(nested_base, d)
                if os.path.isdir(p):
                    return p
    for d in entries:
        if "bird" in d.lower():
            p = os.path.join(base, d)
            if os.path.isdir(p):
                return p
    return cfg_root


def load_species_list(data_root):
    print(f"\n[DATA] data_root = {data_root}")
    for fname in ("taxonomy.csv", "species_list.csv"):
        p = os.path.join(data_root, fname)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            for col in ("primary_label", "species_code", "ebird_code", "inat_taxon_id"):
                if col in df.columns:
                    sp = sorted(df[col].dropna().astype(str).unique().tolist())
                    print(f"[DATA] {len(sp)} species from {fname}[{col}]")
                    return sp
    p = os.path.join(data_root, "sample_submission.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, nrows=1)
        sp = sorted(c for c in df.columns if c.lower() != "row_id")
        if sp:
            print(f"[DATA] {len(sp)} species from sample_submission.csv")
            return sp
    p = os.path.join(data_root, "train.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, usecols=["primary_label"])
        sp = sorted(df["primary_label"].dropna().astype(str).unique().tolist())
        print(f"[DATA] {len(sp)} species from train.csv")
        return sp
    audio_dir = os.path.join(data_root, "train_audio")
    if os.path.isdir(audio_dir):
        sp = sorted(d for d in os.listdir(audio_dir)
                    if os.path.isdir(os.path.join(audio_dir, d)))
        print(f"[DATA] {len(sp)} species from train_audio/ dirs")
        return sp
    raise RuntimeError(f"[DATA] cannot determine species list from {data_root}")


def _parse_hms(t):
    parts = str(t).strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600.0 + m * 60.0 + s


def _parse_secondary(raw, species_to_idx, weight):
    try:
        items = ast.literal_eval(str(raw))
    except Exception:
        return {}
    return {species_to_idx[str(x).strip()]: weight
            for x in items if str(x).strip() in species_to_idx}


def make_label_vector(row, species_to_idx, num_classes, secondary_weight):
    lbl = torch.zeros(num_classes)
    pl = str(row["primary_label"])
    if pl in species_to_idx:
        lbl[species_to_idx[pl]] = 1.0
    for idx, w in _parse_secondary(
            row.get("secondary_labels", "[]"), species_to_idx, secondary_weight).items():
        lbl[idx] = max(lbl[idx].item(), w)
    return lbl

# ── DATASETS ──────────────────────────────────────────────────────────────────

class BirdCLEF2026Dataset(Dataset):
    """Curated species clips from train_audio/ with primary + secondary labels."""

    def __init__(self, data_root, species_to_idx, split="train", duration=5.0,
                 sample_rate=32_000, n_mels=128, n_fft=1024, hop_length=320,
                 val_fraction=0.1, seed=42, secondary_weight=0.5, min_rating=0.0,
                 **_):
        self.audio_dir      = os.path.join(data_root, "train_audio")
        self.species_to_idx = species_to_idx
        self.num_classes    = len(species_to_idx)
        self.split          = split
        self.sample_rate    = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.secondary_weight = secondary_weight

        df = pd.read_csv(os.path.join(data_root, "train.csv"))
        df["primary_label"] = df["primary_label"].astype(str)
        df = df[df["primary_label"].isin(species_to_idx)].copy()
        if min_rating > 0 and "rating" in df.columns:
            df = df[df["rating"].fillna(0.0) >= min_rating].copy()
        print(f"[DATA] clips: {len(df)} ({df['primary_label'].nunique()} sp, split={split})")

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        cut = max(1, int(len(df) * val_fraction))
        self.df = (df.iloc[:cut] if split == "val" else df.iloc[cut:]).reset_index(drop=True)

        self.mel   = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels,
                                      n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.to_db = T.AmplitudeToDB(top_db=80)
        if split == "train":
            self.fmask = T.FrequencyMasking(freq_mask_param=24)
            self.tmask = T.TimeMasking(time_mask_param=80)
        self._resamplers = {}

    def _load_wave(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav

    def _crop_or_pad(self, wav):
        n = wav.shape[-1]
        if n >= self.target_samples:
            start = (torch.randint(0, n - self.target_samples + 1, (1,)).item()
                     if self.split == "train" else (n - self.target_samples) // 2)
            return wav[..., start: start + self.target_samples]
        return F.pad(wav, (0, self.target_samples - n))

    def _to_spec(self, wav):
        spec = self.to_db(self.mel(wav))
        spec = (spec + 80.0) / 80.0
        return spec.clamp(0.0, 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.audio_dir, str(row["filename"]))
        try:
            wav = self._load_wave(path)
        except Exception as e:
            print(f"[WARN] {path}: {e}")
            wav = torch.zeros(1, self.target_samples)
        wav  = self._crop_or_pad(wav)
        spec = self._to_spec(wav)
        if self.split == "train":
            spec = self.fmask(spec)
            spec = self.tmask(spec)
        lbl = make_label_vector(row, self.species_to_idx, self.num_classes,
                                self.secondary_weight)
        return spec, lbl


class SoundscapeDataset(Dataset):
    """Labeled 5-second windows from train_soundscapes/ for training + validation."""

    def __init__(self, data_root, species_to_idx, split="train", duration=5.0,
                 sample_rate=32_000, n_mels=128, n_fft=1024, hop_length=320,
                 val_fraction=0.1, seed=42, **_):
        self.sc_dir       = os.path.join(data_root, "train_soundscapes")
        self.species_to_idx = species_to_idx
        self.num_classes  = len(species_to_idx)
        self.split        = split
        self.sample_rate  = sample_rate
        self.target_len   = int(duration * sample_rate)
        self._n_mels      = n_mels

        lbl_csv = os.path.join(data_root, "train_soundscapes_labels.csv")
        if not os.path.isfile(lbl_csv) or not os.path.isdir(self.sc_dir):
            print("[DATA] SoundscapeDataset: missing — 0 samples")
            self.df = pd.DataFrame(columns=["filename", "start_s", "species_set"])
            self._init_tf(sample_rate, n_mels, n_fft, hop_length)
            return

        df = pd.read_csv(lbl_csv).drop_duplicates(["filename", "start"]).reset_index(drop=True)
        df["start_s"] = df["start"].apply(_parse_hms)
        df["species_set"] = df["primary_label"].apply(
            lambda x: frozenset(s.strip() for s in str(x).split(";")
                                if s.strip() in species_to_idx))
        df = df[df["species_set"].apply(len) > 0].copy()

        files   = sorted(df["filename"].unique())
        rng     = np.random.default_rng(seed)
        rng.shuffle(files)
        val_files = set(files[:max(1, int(len(files) * val_fraction))])
        mask    = df["filename"].isin(val_files)
        self.df = (df[mask] if split == "val" else df[~mask]).reset_index(drop=True)
        print(f"[DATA] soundscapes {split}: {len(self.df)} windows "
              f"({self.df['filename'].nunique()} files)")
        self._init_tf(sample_rate, n_mels, n_fft, hop_length)

    def _init_tf(self, sr, n_mels, n_fft, hop_length):
        self.mel   = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels,
                                      n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.to_db = T.AmplitudeToDB(top_db=80)
        if self.split == "train":
            self.fmask = T.FrequencyMasking(freq_mask_param=24)
            self.tmask = T.TimeMasking(time_mask_param=80)
        self._resamplers = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.sc_dir, row["filename"])
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            print(f"[WARN] {path}: {e}")
            n_t = math.ceil(self.target_len / 320) + 1
            return torch.zeros(1, self._n_mels, n_t), torch.zeros(self.num_classes)

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)

        start = int(row["start_s"] * self.sample_rate)
        end   = start + self.target_len
        if end <= wav.shape[-1]:
            wav = wav[..., start:end]
        elif start < wav.shape[-1]:
            wav = F.pad(wav[..., start:], (0, end - wav.shape[-1]))
        else:
            wav = torch.zeros(1, self.target_len)

        spec = self.to_db(self.mel(wav))
        spec = (spec + 80.0) / 80.0
        spec = spec.clamp(0.0, 1.0)
        if self.split == "train":
            spec = self.fmask(spec)
            spec = self.tmask(spec)

        lbl = torch.zeros(self.num_classes)
        for sp in row["species_set"]:
            if sp in self.species_to_idx:
                lbl[self.species_to_idx[sp]] = 0.75
        return spec, lbl


class TripleClipDataset(Dataset):
    """
    Returns (spec_A, spec_B, spec_bg, spec_mix, lbl_union) for Mode 2 training.
    A, B are clips from different species; bg is a third different species.
    Mix is a weighted sum in spectrogram space (linearized log-power approximation).
    """

    def __init__(self, clip_dataset):
        self.ds = clip_dataset
        by_species = defaultdict(list)
        for i in range(len(clip_dataset)):
            sp = str(clip_dataset.df.iloc[i]["primary_label"])
            by_species[sp].append(i)
        self.by_species = dict(by_species)
        self.species    = list(self.by_species.keys())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        spec_A, lbl_A = self.ds[idx]
        sp_A = str(self.ds.df.iloc[idx]["primary_label"])

        other = [s for s in self.species if s != sp_A]
        sp_B  = random.choice(other) if other else sp_A
        spec_B, lbl_B = self.ds[random.choice(self.by_species[sp_B])]

        other2 = [s for s in self.species if s != sp_A and s != sp_B]
        sp_bg  = random.choice(other2) if other2 else sp_B
        spec_bg, _ = self.ds[random.choice(self.by_species[sp_bg])]

        # Random mix weights (sum to 1)
        w = torch.rand(3) + 0.3
        w = w / w.sum()
        spec_mix = w[0] * spec_A + w[1] * spec_B + w[2] * spec_bg

        lbl_union = (lbl_A + lbl_B).clamp(0.0, 1.0)
        return spec_A, spec_B, spec_bg, spec_mix, lbl_union

# ── MODEL COMPONENTS ──────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # (B, 1, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class ViTBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n1   = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n2   = nn.LayerNorm(d)
        h         = int(d * mlp_ratio)
        self.mlp  = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x    = x + y
        x    = x + self.mlp(self.n2(x))
        return x


class AudioEncoder(nn.Module):
    """
    ViT encoder for log-mel patches.
    forward(x)                → CLS embedding (B, D)
    forward(x, return_tokens) → (CLS (B, D), patch tokens (B, N, D))
    """

    def __init__(self, n_mels=128, n_frames=501, patch_size=16, d=256,
                 n_heads=4, depth=4, dropout=0.1):
        super().__init__()
        n_h = n_mels  // patch_size
        n_w = math.ceil(n_frames / patch_size)
        N   = n_h * n_w

        self.patch_embed = PatchEmbed(d, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed   = nn.Parameter(torch.zeros(1, N + 1, d))
        self.blocks      = nn.ModuleList(
            [ViTBlock(d, n_heads, dropout=dropout) for _ in range(depth)])
        self.norm        = nn.LayerNorm(d)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_tokens=False):  # x: (B, 1, H, W)
        t   = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        t   = torch.cat([cls, t], dim=1)
        t   = t + self.pos_embed[:, :t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        if return_tokens:
            return t[:, 0], t[:, 1:]   # (B, D), (B, N, D)
        return t[:, 0]                  # (B, D)


class SlotAttention(nn.Module):
    """
    Slot Attention (Locatello et al. 2020) with learned slot initialization.

    Softmax over the SLOT dimension (dim=1) — each position "votes" for one slot,
    encouraging mutual exclusivity. Slot updates via GRU + MLP residual.

    Returns: slots (B, K, D), activations (B, K)
      activations = total attention mass per slot (proxy for "how much it used")
    """

    def __init__(self, K=8, D=256, n_iter=3, eps=1e-8):
        super().__init__()
        self.K     = K
        self.D     = D
        self.n_iter = n_iter
        self.eps   = eps
        self.scale = D ** -0.5

        self.slot_mu        = nn.Parameter(torch.zeros(1, 1, D))
        self.slot_log_sigma = nn.Parameter(torch.full((1, 1, D), -2.0))

        self.norm_in    = nn.LayerNorm(D)
        self.norm_slots = nn.LayerNorm(D)
        self.norm_mlp   = nn.LayerNorm(D)

        self.to_k = nn.Linear(D, D, bias=False)
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)

        self.gru = nn.GRUCell(D, D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))

        nn.init.trunc_normal_(self.slot_mu, std=0.02)

    def forward(self, inputs):   # (B, N, D)
        B = inputs.shape[0]

        normed = self.norm_in(inputs)
        k = self.to_k(normed)    # (B, N, D)
        v = self.to_v(normed)    # (B, N, D)

        sigma = self.slot_log_sigma.exp().expand(B, self.K, self.D)
        slots = self.slot_mu.expand(B, self.K, self.D) + sigma * torch.randn_like(sigma)

        for _ in range(self.n_iter):
            prev  = slots
            q     = self.to_q(self.norm_slots(slots))            # (B, K, D)
            dots  = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # (B, K, N)
            attn  = dots.softmax(dim=1)                          # compete over slots
            attn_n = attn / (attn.sum(-1, keepdim=True) + self.eps)  # norm per slot
            upd   = torch.einsum("bkn,bnd->bkd", attn_n, v)     # (B, K, D)
            slots = self.gru(
                upd.reshape(B * self.K, self.D),
                prev.reshape(B * self.K, self.D),
            ).reshape(B, self.K, self.D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        activations = attn.sum(-1)   # (B, K) — total attention mass per slot
        return slots, activations


class ComposePredictor(nn.Module):
    """
    Predicts the composition embedding from slots and their activations.
    Uses softmax-weighted combination of slots → MLP projection.
    Output is NOT unit-normalized (targets are) — BYOL-style collapse prevention.
    """

    def __init__(self, D=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2), nn.GELU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, slots, activations):  # (B, K, D), (B, K)
        w    = activations.softmax(dim=-1).unsqueeze(-1)  # (B, K, 1)
        mixed = (slots * w).sum(1)                        # (B, D)
        return self.mlp(mixed)                            # (B, D)


class SlotClassifier(nn.Module):
    """
    Multi-label classifier with per-slot heads and max pooling.
    Permutation-invariant: species fires if ANY slot detects it.
    Gradient flows from L_cls through all slots.
    """

    def __init__(self, D=256, n_cls=234):
        super().__init__()
        self.head = nn.Linear(D, n_cls)   # shared across slots

    def forward(self, slots):   # (B, K, D) → (B, n_cls)
        per_slot = self.head(slots)                              # (B, K, n_cls)
        return (per_slot.max(dim=1).values + per_slot.mean(dim=1)) * 0.5


class FCJEPAModel(nn.Module):
    """Full Factorize-and-Compose Audio-JEPA model."""

    def __init__(self, n_mels, n_frames, patch_size, D, n_heads, depth, dropout,
                 n_slots, n_slot_iter, n_cls):
        super().__init__()
        enc_args = (n_mels, n_frames, patch_size, D, n_heads, depth, dropout)
        self.enc_ctx = AudioEncoder(*enc_args)
        self.enc_tgt = copy.deepcopy(self.enc_ctx)
        for p in self.enc_tgt.parameters():
            p.requires_grad_(False)

        self.P1       = SlotAttention(n_slots, D, n_slot_iter)
        self.P2       = ComposePredictor(D)
        self.clf      = SlotClassifier(D, n_cls)
        # CLS bypass — preserves step1 classification signal while slots warm up
        self.cls_head = nn.Linear(D, n_cls)
        # Zero-init slot head: starts contributing 0 logits so the CLS head's
        # step1 calibration is fully preserved at epoch 1 and gradually extended.
        nn.init.zeros_(self.clf.head.weight)
        nn.init.zeros_(self.clf.head.bias)

    @torch.no_grad()
    def ema_update(self, tau=0.996):
        for p_c, p_t in zip(self.enc_ctx.parameters(), self.enc_tgt.parameters()):
            p_t.data.mul_(tau).add_((1.0 - tau) * p_c.data)

    @torch.no_grad()
    def encode_target(self, x):
        """Unit-normalized CLS from the EMA target encoder (always stop-grad)."""
        return F.normalize(self.enc_tgt(x), dim=-1)   # (B, D)

    def forward(self, x):
        """Returns (logits, slot_logits, cls_logits, slots, activations, cls).
        Both JEPA losses (L_fact, L_comp, L_div, L_var) and L_cls flow through enc_ctx;
        enc_ctx is kept at 10× lower LR so JEPA objectives dominate the representation."""
        cls, tokens  = self.enc_ctx(x, return_tokens=True)
        slots, acts  = self.P1(tokens)
        slot_logits  = self.clf(slots)
        cls_logits   = self.cls_head(cls)
        logits       = cls_logits + slot_logits
        return logits, slot_logits, cls_logits, slots, acts, cls

    def forward_mix(self, x_mix, x_A, x_B, x_bg):
        """
        Mode 2: mix through context encoder, individuals through target encoder.
        Returns (logits, slot_logits, cls_logits, slots, acts, cls,
                 z_A, z_B, z_bg, z_comp, z_comp_hat).
        """
        cls, tokens  = self.enc_ctx(x_mix, return_tokens=True)
        slots, acts  = self.P1(tokens)
        slot_logits  = self.clf(slots)
        cls_logits   = self.cls_head(cls)
        logits       = cls_logits + slot_logits
        z_comp_hat   = self.P2(slots, acts)

        z_A    = self.encode_target(x_A)
        z_B    = self.encode_target(x_B)
        z_bg   = self.encode_target(x_bg)
        z_comp = F.normalize(z_A + z_B + z_bg, dim=-1)

        return logits, slot_logits, cls_logits, slots, acts, cls, \
               z_A, z_B, z_bg, z_comp, z_comp_hat

# ── LOSSES ────────────────────────────────────────────────────────────────────

def mode1_fact_loss(slots, z_tgt):
    """
    Mode 1 factored loss: assign the closest slot to z_tgt (single target).

    slots:  (B, K, D) — raw (unnormalized) slot vectors
    z_tgt:  (B, D) — unit-normalized target from enc_tgt (stop-grad)

    Cost uses cosine distance on normalized slots (direction only, no magnitude bias).
    Final loss: MSE between normalized matched slot and unit-norm target.
    This is the BYOL formulation — removes the trivial zero-slot minimum that
    exists when comparing unnormalized predictions to unit-norm targets.
    """
    # Cast to float32: autocast converts einsum/mm outputs to bfloat16 inside AMP context
    s_norm = F.normalize(slots.float(), dim=-1)          # (B, K, D)
    z_f    = z_tgt.float()
    cost   = 1.0 - torch.einsum("bkd,bd->bk", s_norm, z_f).float()  # (B, K)
    min_idx = cost.argmin(dim=1)
    matched = s_norm[torch.arange(slots.shape[0]), min_idx]          # (B, D)
    return F.mse_loss(matched, z_f.detach())


def mode2_fact_loss(slots, targets):
    """
    Mode 2 factored loss: Hungarian-matched MSE between K slots and M targets.

    slots:   (B, K, D) — raw slot vectors
    targets: (B, M, D) — unit-normalized targets (M=3: x_A, x_B, x_bg)

    Assignment cost uses cosine distance on normalized slots.
    Final loss: MSE between normalized matched slots and unit-norm targets.
    BYOL formulation — no trivial zero-slot minimum.
    """
    B, K, D = slots.shape
    M       = targets.shape[1]
    s_norm  = F.normalize(slots.float(), dim=-1)   # (B, K, D)
    total   = torch.tensor(0.0, device=slots.device, dtype=slots.dtype)

    for b in range(B):
        sn = s_norm[b]              # (K, D) normalized slots
        t  = targets[b].float()    # (M, D) unit-norm targets
        # (K, M) cosine distance cost matrix — cast to float32 before numpy (bfloat16 unsupported)
        cost = 1.0 - torch.mm(sn, t.t())
        row_ind, col_ind = linear_sum_assignment(cost.detach().float().cpu().numpy())
        total = total + F.mse_loss(sn[row_ind], t[col_ind].detach())

    return total / B


def diversity_loss(slots):
    """
    Slot diversity: penalize off-diagonal cosine similarity in slot Gram matrix.
    Encourages each slot to specialize for a distinct acoustic pattern.

    slots: (B, K, D) → scalar loss (higher = more similar = bad)
    """
    K = slots.shape[1]
    s = F.normalize(slots, dim=-1)                          # (B, K, D)
    gram = torch.bmm(s, s.transpose(1, 2))                  # (B, K, K)
    mask = ~torch.eye(K, dtype=torch.bool, device=slots.device)
    return gram[:, mask].clamp(min=0.0).mean()


def variance_loss(z, slots=None, target_std=1.0):
    """
    VICReg variance: penalize CLS dimensions with std < target_std across the batch.
    Slot anti-collapse is handled by diversity_loss (off-diagonal cosine) so we
    don't double-penalize here — conflicting slot gradients cause JEPA loss collapse.

    z:     (B, D) — context encoder CLS output
    slots: unused, kept for call-site compatibility
    """
    return F.relu(target_std - z.float().std(dim=0)).mean()


# ── LR SCHEDULE ───────────────────────────────────────────────────────────────

def cosine_lr(epoch, total, warmup, base, floor=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * t))

# ── STEP 1 WARM START ─────────────────────────────────────────────────────────

def load_step1_weights(model, ckpt_path, device):
    """
    Load enc_ctx and cls_head from a Step 1 BaselineClassifier checkpoint.
    Step 1 saves as: enc.* (AudioEncoder) + head.* (Linear classifier).
    cls_head gets the step1 head → preserves classification signal at epoch 0.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    enc_state  = {k[4:]: v for k, v in state.items() if k.startswith("enc.")}
    head_state = {k[5:]: v for k, v in state.items() if k.startswith("head.")}
    m1, u1 = model.enc_ctx.load_state_dict(enc_state, strict=False)
    m2, u2 = model.cls_head.load_state_dict(head_state, strict=False)
    model.enc_tgt.load_state_dict(model.enc_ctx.state_dict())
    print(f"[CKPT] enc_ctx missing={len(m1)} unexpected={len(u1)}  |  "
          f"cls_head missing={len(m2)} unexpected={len(u2)}")
    return ckpt.get("species", None)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--out_dir",     default=None)
    parser.add_argument("--step1_ckpt",  default=None,
                        help="Path to Step 1 best.pt for warm start")
    parser.add_argument("--no_mix",      action="store_true",
                        help="Disable Mode 2 (synthetic mix) training")
    args, _ = parser.parse_known_args()

    cfg = dict(CFG)
    if args.data_root:
        cfg["data_root"] = args.data_root
    elif os.environ.get("BIRDCLEF_DATA_ROOT"):
        cfg["data_root"] = os.environ["BIRDCLEF_DATA_ROOT"]
    if args.step1_ckpt:
        cfg["step1_ckpt"] = args.step1_ckpt
    elif os.environ.get("STEP1_CKPT"):
        cfg["step1_ckpt"] = os.environ["STEP1_CKPT"]
    if args.no_mix:
        cfg["mix_every"] = 0

    out_dir = args.out_dir or os.environ.get("BIRDCLEF_OUT_DIR", "/kaggle/working")
    os.makedirs(out_dir, exist_ok=True)

    root = find_data_root(cfg["data_root"])
    print("─" * 60)
    print("FC Audio JEPA — Step 3: Factorize-and-Compose JEPA")
    print("─" * 60)

    species_list   = load_species_list(root)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes    = len(species_list)
    print(f"[MODEL] species: {num_classes}")

    ds_kwargs = dict(
        data_root      = root,
        species_to_idx = species_to_idx,
        duration       = cfg["duration"],
        sample_rate    = cfg["sample_rate"],
        n_mels         = cfg["n_mels"],
        n_fft          = cfg["n_fft"],
        hop_length     = cfg["hop_length"],
        val_fraction   = cfg["val_fraction"],
        seed           = cfg["seed"],
    )

    sc_tr = SoundscapeDataset(split="train", **ds_kwargs)
    sc_va = SoundscapeDataset(split="val",   **ds_kwargs)
    cl_tr = BirdCLEF2026Dataset(split="train", secondary_weight=cfg["secondary_weight"],
                                min_rating=cfg["min_rating"], **ds_kwargs)
    cl_va = BirdCLEF2026Dataset(split="val",   secondary_weight=cfg["secondary_weight"],
                                min_rating=cfg["min_rating"], **ds_kwargs)

    train_ds = ConcatDataset([sc_tr, cl_tr]) if len(sc_tr) > 0 else cl_tr
    if len(sc_va) >= 200:
        val_ds = sc_va
    else:
        val_ds = ConcatDataset([sc_va, cl_va]) if len(sc_va) > 0 else cl_va
        print(f"[DATA] val top-up: sc_va={len(sc_va)}, adding {len(cl_va)} clip val")
    print(f"[DATA] train={len(train_ds):,}  val={len(val_ds):,}")

    # ── GPU + precision + batch size (MUST precede DataLoader) ────────────────
    n_frames = math.ceil(cfg["duration"] * cfg["sample_rate"] / cfg["hop_length"]) + 1
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        cap     = torch.cuda.get_device_capability()
        cap_val = cap[0] + cap[1] / 10
        amp_dtype = (torch.bfloat16 if cap_val >= 8.0 else
                     torch.float16  if cap_val >= 7.0 else torch.float32)
        print(f"[MODEL] device={device}  cap=sm_{cap[0]}{cap[1]}  "
              f"amp={amp_dtype}  spec=({cfg['n_mels']}, {n_frames})")
    else:
        cap_val = 0.0
        amp_dtype = torch.float32
        print(f"[MODEL] device={device}  spec=({cfg['n_mels']}, {n_frames})")

    use_amp = (amp_dtype != torch.float32)
    scaler  = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    cfg = dict(cfg)
    if device.type == "cpu":
        cfg["epochs"]         = 2
        cfg["batch_size"]     = 16
        cfg["mix_batch_size"] = 8
        cfg["num_workers"]    = 0
        cfg["mix_every"]      = 0
        print("[TRAIN] CPU smoke-test mode — epochs=2, batch_size=16, no Mode 2")
    elif cap_val >= 8.0:
        cfg["batch_size"]     = 256
        cfg["mix_batch_size"] = 128
        cfg["num_workers"]    = 12
        print(f"[TRAIN] A100 mode — batch={cfg['batch_size']}  "
              f"mix_batch={cfg['mix_batch_size']}  workers={cfg['num_workers']}")

    steps_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    print(f"[TRAIN] steps/epoch={steps_per_epoch}  "
          f"total={steps_per_epoch * cfg['epochs']}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)

    # Mode 2 loader (TripleClipDataset from training clips only)
    mix_loader = None
    mix_iter   = None
    if cfg["mix_every"] > 0:
        mix_ds     = TripleClipDataset(cl_tr)
        mix_loader = DataLoader(mix_ds, batch_size=cfg["mix_batch_size"],
                                shuffle=True, num_workers=cfg["num_workers"],
                                pin_memory=True, drop_last=True)
        mix_iter   = iter(mix_loader)
        print(f"[TRAIN] Mode 2 enabled: mix_every={cfg['mix_every']}, "
              f"mix_ds={len(mix_ds):,}")
    else:
        print("[TRAIN] Mode 2 disabled (mix_every=0)")

    # ── WandB run ─────────────────────────────────────────────────────────────
    if _USE_WANDB:
        wandb.init(project="fc-audio-jepa",
                   name=f"step3-fcjepa-{device.type}",
                   config=cfg,
                   tags=["step3", "jepa", "slots", "birdclef2026"])
        print("[WANDB] run started")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FCJEPAModel(
        n_mels    = cfg["n_mels"],
        n_frames  = n_frames,
        patch_size = cfg["patch_size"],
        D         = cfg["d_model"],
        n_heads   = cfg["n_heads"],
        depth     = cfg["n_layers"],
        dropout   = cfg["dropout"],
        n_slots   = cfg["n_slots"],
        n_slot_iter = cfg["n_slot_iter"],
        n_cls     = num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable params: {n_params / 1e6:.2f}M  "
          f"(enc_tgt frozen — EMA only)")

    # Warm start from Step 1 checkpoint
    if cfg["step1_ckpt"] and os.path.isfile(cfg["step1_ckpt"]):
        load_step1_weights(model, cfg["step1_ckpt"], device)
    else:
        if cfg["step1_ckpt"]:
            print(f"[WARN] step1_ckpt={cfg['step1_ckpt']} not found — training from scratch")
        else:
            print("[TRAIN] No step1_ckpt — training from scratch")

    # Class-frequency positive weight
    cc = torch.zeros(num_classes)
    for sp in cl_tr.df["primary_label"].astype(str):
        if sp in species_to_idx:
            cc[species_to_idx[sp]] += 1
    pos_weight = ((len(cl_tr) - cc) / cc.clamp(min=1)).clamp(1.0, 50.0).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential LR: encoder (already trained) gets 10× lower rate than new heads
    enc_lr_scale = cfg.get("encoder_lr_scale", 0.1)
    enc_params   = [p for p in model.enc_ctx.parameters() if p.requires_grad]
    new_params   = (list(model.P1.parameters()) + list(model.P2.parameters()) +
                    list(model.clf.parameters()) + list(model.cls_head.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["lr"] * enc_lr_scale, "name": "encoder"},
        {"params": new_params, "lr": cfg["lr"],                 "name": "heads"},
    ], weight_decay=cfg["weight_decay"])
    print(f"[TRAIN] LR: encoder={cfg['lr']*enc_lr_scale:.1e}  "
          f"new_heads={cfg['lr']:.1e}")

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc    = 0.0
    global_step = 0
    log_every   = cfg.get("log_every", 1)

    lf = cfg["lambda_fact"]
    lc = cfg["lambda_comp"]
    ld = cfg["lambda_div"]
    lv = cfg["lambda_var"]

    for ep in range(cfg["epochs"]):
        lr     = cosine_lr(ep, cfg["epochs"], cfg["warmup_epochs"], cfg["lr"])
        enc_lr = cosine_lr(ep, cfg["epochs"], cfg["warmup_epochs"],
                           cfg["lr"] * cfg.get("encoder_lr_scale", 0.1))
        # End-to-end training: both param groups active from epoch 1.
        # Differential LR keeps enc_ctx from diverging away from step1 features.
        optimizer.param_groups[0]["lr"] = enc_lr   # enc_ctx (10× lower)
        optimizer.param_groups[1]["lr"] = lr        # P1, P2, clf, cls_head

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        ep_cls = ep_fact = ep_comp = ep_div = ep_var = 0.0
        ep_n_mode2 = 0
        nb = 0

        for specs, lbls in train_loader:
            specs = specs.to(device)
            lbls  = lbls.to(device)
            optimizer.zero_grad()

            mode2_triggered = (mix_loader is not None
                               and global_step % cfg["mix_every"] == 0)

            if mode2_triggered:
                # ── Mode 2 step ───────────────────────────────────────────────
                try:
                    m2 = next(mix_iter)
                except StopIteration:
                    mix_iter = iter(mix_loader)
                    m2 = next(mix_iter)

                sA, sB, sbg, smix, lbl_u = [t.to(device) for t in m2]

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    (logits, slot_logits, cls_logits, slots, acts, cls,
                     z_A, z_B, z_bg, z_comp, z_comp_hat) = model.forward_mix(
                        smix, sA, sB, sbg)

                    targets = torch.stack([z_A, z_B, z_bg], dim=1)  # (B, 3, D)
                    L_cls  = criterion(logits, lbl_u)
                    L_fact = mode2_fact_loss(slots, targets)
                    # Normalize z_comp_hat before MSE — BYOL formulation.
                    # Prevents trivial near-zero P2 output minimizing loss vs unit target.
                    L_comp = F.mse_loss(F.normalize(z_comp_hat.float(), dim=-1),
                                        z_comp.detach())
                    L_div  = diversity_loss(slots)
                    L_var  = variance_loss(cls, slots)
                    loss   = (L_cls
                              + lf * L_fact
                              + lc * L_comp
                              + ld * L_div
                              + lv * L_var)

                log_dict = {
                    "train/L_cls":  L_cls.item(),
                    "train/L_fact": L_fact.item(),
                    "train/L_comp": L_comp.item(),
                    "train/L_div":  L_div.item(),
                    "train/L_var":  L_var.item(),
                    "train/mode":   2,
                }
                ep_cls  += L_cls.item()
                ep_fact += L_fact.item()
                ep_comp += L_comp.item()
                ep_div  += L_div.item()
                ep_var  += L_var.item()
                ep_n_mode2 += 1
            else:
                # ── Mode 1 step ───────────────────────────────────────────────
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    logits, slot_logits, cls_logits, slots, acts, cls = model(specs)
                    z_tgt  = model.encode_target(specs)   # stop-grad, unit-normed
                    L_cls  = criterion(logits, lbls)
                    L_fact = mode1_fact_loss(slots, z_tgt)
                    L_div  = diversity_loss(slots)
                    L_var  = variance_loss(cls, slots)
                    loss   = (L_cls
                              + lf * L_fact
                              + ld * L_div
                              + lv * L_var)

                log_dict = {
                    "train/L_cls":  L_cls.item(),
                    "train/L_fact": L_fact.item(),
                    "train/L_div":  L_div.item(),
                    "train/L_var":  L_var.item(),
                    "train/mode":   1,
                }
                ep_cls  += L_cls.item()
                ep_fact += L_fact.item()
                ep_div  += L_div.item()
                ep_var  += L_var.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            model.ema_update(cfg["ema_tau"])

            ep_loss += loss.item()
            nb      += 1
            global_step += 1

            if _USE_WANDB and global_step % log_every == 0:
                log_dict["train/grad_norm"] = grad_norm
                log_dict["train/lr_heads"]  = lr
                log_dict["train/lr_enc"]    = enc_lr
                log_dict["step"]            = global_step
                wandb.log(log_dict)

        ep_loss /= max(nb, 1)
        ep_cls  /= max(nb, 1)
        ep_fact /= max(nb, 1)
        ep_comp /= max(ep_n_mode2, 1)   # only Mode 2 steps have L_comp
        ep_div  /= max(nb, 1)
        ep_var  /= max(nb, 1)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        pall, lall = [], []
        with torch.no_grad():
            for specs, lbls in val_loader:
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    logits, _, _, _, _, _ = model(specs.to(device))
                pred = torch.sigmoid(logits).cpu().float()
                pall.append(pred.numpy())
                lall.append(lbls.float().numpy())

        pall = np.concatenate(pall)
        lall = np.concatenate(lall)
        # Threshold at >= 0.5 so secondary labels (stored as 0.5) count as present.
        # Matches Kaggle's metric: scored_columns = solution.sum(axis=0) > 0
        # (strict > 0.5 would silently drop secondary-only species from scoring).
        lall_hard = (lall >= 0.5).astype(float)
        col_sums  = lall_hard.sum(0)
        valid     = col_sums > 0           # Kaggle: sums[sums > 0]
        val_auc   = (roc_auc_score(lall_hard[:, valid], pall[:, valid], average="macro")
                     if valid.sum() > 0 else 0.0)
        n_valid   = int(valid.sum())
        mean_pred = float(pall.mean())

        m2_frac = ep_n_mode2 / max(nb, 1)
        print(f"Epoch {ep+1:02d}/{cfg['epochs']:02d} | "
              f"loss {ep_loss:.4f} | "
              f"cls {ep_cls:.4f} | fact {ep_fact:.4f} | "
              f"comp {ep_comp:.4f} | div {ep_div:.4f} | var {ep_var:.4f} | "
              f"m2 {m2_frac:.2f} | "
              f"val_auc {val_auc:.4f} | mean_pred {mean_pred:.3f}")

        if _USE_WANDB:
            wandb.log({
                "train/epoch_loss":   ep_loss,
                "train/epoch_L_cls":  ep_cls,
                "train/epoch_L_fact": ep_fact,
                "train/epoch_L_comp": ep_comp,
                "train/epoch_L_div":  ep_div,
                "train/epoch_L_var":  ep_var,
                "train/mode2_frac":   m2_frac,
                "val/auc":            val_auc,
                "val/species":        n_valid,
                "val/mean_pred":      mean_pred,
                "epoch":              ep + 1,
            })

        payload = {"epoch": ep + 1, "model_state": model.state_dict(),
                   "val_auc": val_auc, "cfg": cfg, "species": species_list}
        torch.save(payload, f"{ckpt_dir}/last.pt")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(payload, f"{ckpt_dir}/best.pt")
            print(f"  ↑ best AUC: {best_auc:.4f}")

    if _USE_WANDB:
        wandb.summary["best_val_auc"] = best_auc
        wandb.finish()
    print(f"\nBest val AUC: {best_auc:.4f}  |  ckpt: {ckpt_dir}/best.pt")

    # ── Inference ─────────────────────────────────────────────────────────────
    ckpt_path = (f"{ckpt_dir}/best.pt" if os.path.exists(f"{ckpt_dir}/best.pt")
                 else f"{ckpt_dir}/last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    run_inference(model, cfg, root, species_list, species_to_idx, device, out_dir)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, cfg, data_root, species_list, species_to_idx, device,
                  out_dir="/kaggle/working"):
    import glob
    model.eval()
    sr        = cfg["sample_rate"]
    n_samples = int(sr * cfg["duration"])
    win_sec   = int(cfg["duration"])

    mel_tf = T.MelSpectrogram(sample_rate=sr, n_mels=cfg["n_mels"],
                              n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
    db_tf  = T.AmplitudeToDB(top_db=80)

    sub_tmpl    = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))
    species_cols = [c for c in sub_tmpl.columns if c != "row_id"]
    col_idx      = {col: species_to_idx.get(col, -1) for col in species_cols}
    print(f"[INFER] {len(sub_tmpl)} rows, {len(species_cols)} species cols")

    preds    = {}
    test_dir = os.path.join(data_root, "test_soundscapes")
    files    = sorted(glob.glob(os.path.join(test_dir, "*.ogg")))
    print(f"[INFER] {len(files)} test soundscape files")

    with torch.no_grad():
        for fpath in files:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            wav, file_sr = torchaudio.load(fpath)
            if file_sr != sr:
                wav = T.Resample(file_sr, sr)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)

            start   = 0
            end_sec = win_sec
            while start < wav.shape[1]:
                chunk = wav[:, start: start + n_samples]
                if chunk.shape[1] < n_samples:
                    chunk = F.pad(chunk, (0, n_samples - chunk.shape[1]))
                spec = db_tf(mel_tf(chunk))
                spec = (spec + 80.0) / 80.0
                spec = spec.clamp(0.0, 1.0).unsqueeze(0).to(device)

                logits, _, _, _, _, _ = model(spec)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                preds[f"{stem}_{end_sec}"] = probs

                start   += n_samples
                end_sec += win_sec

    rows = []
    for rid in sub_tmpl["row_id"]:
        p   = preds.get(rid, np.zeros(len(species_list), dtype=np.float32))
        row = {"row_id": rid}
        for col in species_cols:
            idx = col_idx[col]
            row[col] = float(p[idx]) if idx >= 0 else 0.0
        rows.append(row)

    sub_df   = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"[INFER] submission.csv → {out_path}  "
          f"({len(sub_df)} rows × {len(species_cols)} cols)")


if __name__ == "__main__":
    main()
