"""
Step 1: Supervised baseline — scratch ViT + multi-label BCE on BirdCLEF 2026.

BUNDLED: all dataset, model, and training code in one file (Kaggle script mode).

BirdCLEF 2026 data schema
─────────────────────────
  taxonomy.csv              : primary_label (string), inat_taxon_id, ...
                              primary_label is the species ID used everywhere:
                                • numeric iNat taxon ID  (non-birds)
                                • eBird code             (birds: "compau", ...)
  train.csv                 : metadata for curated clips — primary_label,
                              secondary_labels (Python list string), filename
                              ("taxon_id/file.ogg"), rating, ...
  train_audio/              : train_audio/{primary_label}/{clip}.ogg
  train_soundscapes/        : passive field recordings (matches test dist.)
  train_soundscapes_labels.csv : filename, start (HH:MM:SS), end, primary_label
                              where primary_label = semicolon-separated species IDs

  ▸ Both datasets are combined for training.
  ▸ Validation uses soundscape windows only (closest to test distribution).
  ▸ Val split is by soundscape FILE (not window) to prevent leakage.

Labels
──────
  Soundscape windows : all semicolon-separated IDs → hard positive (1.0)
  Curated clips      : primary_label → 1.0 | secondary_labels → 0.5
"""

import ast
import math
import os
import sys

# ── CUDA compatibility guard ───────────────────────────────────────────────────
# Modern PyTorch (≥2.3) dropped support for CUDA capability < 7.0 (e.g. P100).
# Detect this BEFORE importing torch.cuda to avoid silent fallback crashes.
try:
    import subprocess
    _nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    _caps = [float(c.strip()) for c in _nvsmi.stdout.strip().splitlines() if c.strip()]
    if _caps and all(c < 7.0 for c in _caps):
        print(f"[GPU] CUDA capability {_caps} < 7.0 — forcing CPU mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Audio frontend
    sample_rate   = 32_000,
    n_mels        = 128,
    n_fft         = 1024,
    hop_length    = 320,
    duration      = 5.0,       # seconds → ~501 time frames at these settings
    # Model
    d_model       = 256,
    n_heads       = 4,
    n_layers      = 4,
    patch_size    = 16,
    dropout       = 0.1,
    # Training
    epochs        = 15,
    batch_size    = 64,
    lr            = 1e-3,
    weight_decay  = 0.05,
    warmup_epochs = 2,
    # Data
    data_root        = "/kaggle/input/birdclef-2026",
    num_workers      = 4,
    val_fraction     = 0.1,
    seed             = 42,
    secondary_weight = 0.5,    # label weight for secondary (uncertain) labels
    min_rating       = 0.0,    # keep all recordings (set > 0 to filter low quality)
)

# ── SPECIES LIST ──────────────────────────────────────────────────────────────

def find_data_root(cfg_root: str = "/kaggle/input/birdclef-2026") -> str:
    """
    Return the actual data root, scanning /kaggle/input/ if the configured
    path doesn't exist.

    Kaggle competition data can be mounted at:
      /kaggle/input/birdclef-2026/          (direct)
      /kaggle/input/competitions/birdclef-2026/  (nested under competitions/)

    Prints debug output so mount failures are visible in the kernel log.
    """
    if os.path.isdir(cfg_root):
        return cfg_root

    base = "/kaggle/input"
    print(f"[DATA] '{cfg_root}' not found — scanning {base}")
    if not os.path.isdir(base):
        print(f"[DATA] {base} does not exist — is this running outside Kaggle?")
        return cfg_root

    entries = sorted(os.listdir(base))
    print(f"[DATA] /kaggle/input/ contents: {entries}")

    # Kaggle sometimes nests competition data under /kaggle/input/competitions/
    nested_base = os.path.join(base, "competitions")
    if os.path.isdir(nested_base):
        nested = sorted(os.listdir(nested_base))
        print(f"[DATA] /kaggle/input/competitions/: {nested}")
        for d in nested:
            if "birdclef" in d.lower() or "bird" in d.lower():
                path = os.path.join(nested_base, d)
                if os.path.isdir(path):
                    print(f"[DATA] → using {path}")
                    return path
        # Take first entry under competitions/
        for d in nested:
            path = os.path.join(nested_base, d)
            if os.path.isdir(path):
                print(f"[DATA] → falling back to {path}")
                return path

    # Search top-level entries for birdclef/bird
    for d in entries:
        if "birdclef" in d.lower() or "bird" in d.lower():
            path = os.path.join(base, d)
            if os.path.isdir(path):
                print(f"[DATA] → using {path}")
                return path

    # Last resort: first available top-level directory
    for d in entries:
        path = os.path.join(base, d)
        if os.path.isdir(path):
            print(f"[DATA] → last resort: {path}")
            return path

    return cfg_root  # will fail later with a clear message


def load_species_list(data_root: str) -> list[str]:
    """
    Return sorted list of species primary_labels. Tries four strategies:
      1. taxonomy.csv  primary_label column (most reliable)
      2. sample_submission.csv columns minus row_id (always present in competitions)
      3. train.csv     primary_label column
      4. train_audio/  subdirectory names

    Prints a brief [DATA] debug line for each attempt so failures are visible
    in the Kaggle log.
    """
    print(f"\n[DATA] data_root = {data_root}")
    try:
        items = sorted(os.listdir(data_root))
        print(f"[DATA] root contents ({len(items)} items): {items[:25]}")
    except Exception as e:
        print(f"[DATA] cannot list root: {e}")

    # ── 1. taxonomy.csv ───────────────────────────────────────────────────────
    for fname in ("taxonomy.csv", "species_list.csv"):
        p = os.path.join(data_root, fname)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            print(f"[DATA] {fname} shape={df.shape}  cols={list(df.columns)}")
            for col in ("primary_label", "species_code", "ebird_code", "inat_taxon_id"):
                if col in df.columns:
                    sp = sorted(df[col].dropna().astype(str).unique().tolist())
                    print(f"[DATA] → {len(sp)} species from {fname}[{col}]")
                    return sp

    # ── 2. sample_submission.csv columns ─────────────────────────────────────
    p = os.path.join(data_root, "sample_submission.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, nrows=1)
        sp = sorted(c for c in df.columns if c.lower() != "row_id")
        if sp:
            print(f"[DATA] → {len(sp)} species from sample_submission.csv columns")
            return sp

    # ── 3. train.csv primary_label ────────────────────────────────────────────
    p = os.path.join(data_root, "train.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, usecols=["primary_label"])
        sp = sorted(df["primary_label"].dropna().astype(str).unique().tolist())
        print(f"[DATA] → {len(sp)} species from train.csv primary_label")
        return sp

    # ── 4. train_audio/ subdirectory names ───────────────────────────────────
    audio_dir = os.path.join(data_root, "train_audio")
    if os.path.isdir(audio_dir):
        sp = sorted(
            d for d in os.listdir(audio_dir)
            if os.path.isdir(os.path.join(audio_dir, d))
        )
        print(f"[DATA] → {len(sp)} species from train_audio/ dirs")
        return sp

    raise RuntimeError(
        f"[DATA] Cannot determine species list from {data_root}.\n"
        "See [DATA] lines above for available files."
    )


# ── AUDIO / LABEL HELPERS ─────────────────────────────────────────────────────

def _parse_hms(t: str) -> float:
    """Parse 'HH:MM:SS' to float seconds."""
    parts = str(t).strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600.0 + m * 60.0 + s


def _parse_secondary(raw, species_to_idx: dict, weight: float) -> dict:
    """
    Parse secondary_labels field (Python list literal string) into
    {species_idx: weight} pairs.

    Examples of raw values from train.csv:
      "[]"               → {}
      "['compau']"       → {idx_compau: weight}
      "['22930', '116570']" → {idx_22930: weight, idx_116570: weight}
    """
    try:
        items = ast.literal_eval(str(raw))
    except Exception:
        return {}
    out = {}
    for item in items:
        key = str(item).strip()
        if key in species_to_idx:
            out[species_to_idx[key]] = weight
    return out


def make_label_vector(row, species_to_idx: dict, num_classes: int,
                      secondary_weight: float) -> torch.Tensor:
    lbl = torch.zeros(num_classes)
    # Primary label — hard positive
    pl = str(row["primary_label"])
    if pl in species_to_idx:
        lbl[species_to_idx[pl]] = 1.0
    # Secondary labels — soft positives (uncertain co-occurrences)
    for idx, w in _parse_secondary(row.get("secondary_labels", "[]"),
                                   species_to_idx, secondary_weight).items():
        lbl[idx] = max(lbl[idx].item(), w)
    return lbl


# ── DATASET ───────────────────────────────────────────────────────────────────

class BirdCLEF2026Dataset(Dataset):
    """
    Loads curated species clips from train_audio/ using train.csv metadata.

    Each clip may have a primary species (hard positive) and secondary species
    (soft positives, weight=secondary_weight). Audio is sliced to fixed-length
    windows and converted to log-mel spectrograms.

    train_audio/{primary_label}/{clip}.ogg — but the `filename` column in
    train.csv already gives "{primary_label}/{clip}.ogg" relative to train_audio/.
    """

    def __init__(
        self,
        data_root: str,
        species_to_idx: dict,
        split: str = "train",
        duration: float = 5.0,
        sample_rate: int = 32_000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        val_fraction: float = 0.1,
        seed: int = 42,
        secondary_weight: float = 0.5,
        min_rating: float = 0.0,
    ):
        self.data_root      = data_root
        self.audio_dir      = os.path.join(data_root, "train_audio")
        self.species_to_idx = species_to_idx
        self.num_classes    = len(species_to_idx)
        self.split          = split
        self.sample_rate    = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.secondary_weight = secondary_weight

        # Load metadata
        train_csv = os.path.join(data_root, "train.csv")
        df = pd.read_csv(train_csv)
        df["primary_label"] = df["primary_label"].astype(str)

        # Keep only species that are in our species list
        df = df[df["primary_label"].isin(species_to_idx)].copy()

        # Filter by rating if requested
        if min_rating > 0 and "rating" in df.columns:
            df = df[df["rating"].fillna(0.0) >= min_rating].copy()

        print(f"[DATA] train.csv: {len(df)} clips retained  "
              f"({df['primary_label'].nunique()} species, split={split})")

        # Train / val split — simple random (stratified would be better)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        cut = max(1, int(len(df) * val_fraction))
        self.df = (df.iloc[:cut] if split == "val" else df.iloc[cut:]).reset_index(drop=True)

        # Audio transforms
        self.mel   = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
        if split == "train":
            self.fmask = T.FrequencyMasking(freq_mask_param=24)
            self.tmask = T.TimeMasking(time_mask_param=80)
        self._resamplers: dict = {}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_wave(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav  # (1, N)

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n >= self.target_samples:
            start = (
                torch.randint(0, n - self.target_samples + 1, (1,)).item()
                if self.split == "train"
                else (n - self.target_samples) // 2
            )
            return wav[..., start: start + self.target_samples]
        return F.pad(wav, (0, self.target_samples - n))

    # ── public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Audio path: train_audio/{filename}  e.g. "1161364/iNat1114648.ogg"
        path = os.path.join(self.audio_dir, str(row["filename"]))
        try:
            wav = self._load_wave(path)
        except Exception as e:
            print(f"[WARN] failed to load {path}: {e} — using silence")
            wav = torch.zeros(1, self.target_samples)

        wav  = self._crop_or_pad(wav)
        spec = self.mel(wav)
        spec = self.to_db(spec)
        spec = (spec + 80.0) / 80.0
        spec = spec.clamp(0.0, 1.0)

        if self.split == "train":
            spec = self.fmask(spec)
            spec = self.tmask(spec)

        lbl = make_label_vector(
            row, self.species_to_idx, self.num_classes, self.secondary_weight
        )
        return spec, lbl


# ── SOUNDSCAPE DATASET ────────────────────────────────────────────────────────

class SoundscapeDataset(Dataset):
    """
    Loads labeled 5-second windows from train_soundscapes/ using
    train_soundscapes_labels.csv.

    CSV schema:
      filename     – soundscape OGG file (e.g. BC2026_Train_0039_...ogg)
      start        – window start HH:MM:SS
      end          – window end   HH:MM:SS
      primary_label – semicolon-separated species IDs present in window

    Val split is by soundscape FILE (not by window) to prevent leakage.
    These recordings match the hidden test-set distribution, making this
    dataset the primary supervision signal for the supervised baseline and
    the context-encoder (E_c) input in later JEPA stages.
    """

    def __init__(
        self,
        data_root: str,
        species_to_idx: dict,
        split: str = "train",
        duration: float = 5.0,
        sample_rate: int = 32_000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        val_fraction: float = 0.1,
        seed: int = 42,
        **_kwargs,  # absorb unused kwargs from shared ds_kwargs dict
    ):
        self.sc_dir       = os.path.join(data_root, "train_soundscapes")
        self.species_to_idx = species_to_idx
        self.num_classes  = len(species_to_idx)
        self.split        = split
        self.sample_rate  = sample_rate
        self.target_len   = int(duration * sample_rate)

        lbl_csv = os.path.join(data_root, "train_soundscapes_labels.csv")
        if not os.path.isfile(lbl_csv) or not os.path.isdir(self.sc_dir):
            print(f"[DATA] SoundscapeDataset: missing labels or dir — 0 samples")
            self.df = pd.DataFrame(columns=["filename", "start_s", "species_set"])
            self._init_transforms(sample_rate, n_mels, n_fft, hop_length)
            return

        df = pd.read_csv(lbl_csv)
        df = df.drop_duplicates(subset=["filename", "start"]).reset_index(drop=True)
        df["start_s"] = df["start"].apply(_parse_hms)
        self._n_mels = n_mels
        # Parse semicolon-separated label string → frozenset of known IDs
        df["species_set"] = df["primary_label"].apply(
            lambda x: frozenset(
                s.strip() for s in str(x).split(";")
                if s.strip() in species_to_idx
            )
        )
        df = df[df["species_set"].apply(len) > 0].copy()

        # Split by FILE (prevent window-level leakage)
        files = sorted(df["filename"].unique())
        rng   = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val     = max(1, int(len(files) * val_fraction))
        val_files = set(files[:n_val])

        mask    = df["filename"].isin(val_files)
        self.df = (df[mask] if split == "val" else df[~mask]).reset_index(drop=True)

        print(f"[DATA] SoundscapeDataset split={split}: {len(self.df)} windows  "
              f"({self.df['filename'].nunique()} files)")

        self._init_transforms(sample_rate, n_mels, n_fft, hop_length)

    def _init_transforms(self, sample_rate, n_mels, n_fft, hop_length):
        self._n_mels = n_mels
        self.mel   = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
        if self.split == "train":
            self.fmask = T.FrequencyMasking(freq_mask_param=24)
            self.tmask = T.TimeMasking(time_mask_param=80)
        self._resamplers: dict = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.sc_dir, row["filename"])
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            print(f"[WARN] {path}: {e}")
            n_frames = math.ceil(self.target_len / 320) + 1
            return torch.zeros(1, self._n_mels, n_frames), torch.zeros(self.num_classes)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)

        # Extract fixed-length window from start_s
        start = int(row["start_s"] * self.sample_rate)
        end   = start + self.target_len
        if end <= wav.shape[-1]:
            wav = wav[..., start:end]
        elif start < wav.shape[-1]:
            wav = F.pad(wav[..., start:], (0, end - wav.shape[-1]))
        else:
            wav = torch.zeros(1, self.target_len)

        spec = self.mel(wav)
        spec = self.to_db(spec)
        spec = (spec + 80.0) / 80.0
        spec = spec.clamp(0.0, 1.0)

        if self.split == "train":
            spec = self.fmask(spec)
            spec = self.tmask(spec)

        lbl = torch.zeros(self.num_classes)
        for sp in row["species_set"]:
            if sp in self.species_to_idx:
                lbl[self.species_to_idx[sp]] = 1.0
        return spec, lbl


# ── MODEL ─────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """2-D patch embedding for single-channel log-mel spectrograms."""

    def __init__(self, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):           # x: (B, 1, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)


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

    def forward(self, x):
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x    = x + y
        x    = x + self.mlp(self.n2(x))
        return x


class AudioEncoder(nn.Module):
    """Scratch ViT encoder for log-mel patches. Returns CLS-token embedding."""

    def __init__(self, n_mels: int = 128, n_frames: int = 501,
                 patch_size: int = 16, d: int = 256,
                 n_heads: int = 4, depth: int = 4, dropout: float = 0.1):
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

    def forward(self, x):           # x: (B, 1, n_mels, n_frames)
        t = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        t   = torch.cat([cls, t], dim=1)
        t   = t + self.pos_embed[:, :t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)[:, 0]   # CLS → (B, D)


class BaselineClassifier(nn.Module):
    def __init__(self, n_mels, n_frames, patch_size, d,
                 n_heads, depth, dropout, n_cls):
        super().__init__()
        self.enc  = AudioEncoder(n_mels, n_frames, patch_size, d, n_heads, depth, dropout)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        return self.head(self.enc(x))   # logits (B, n_cls)


# ── LR SCHEDULE ───────────────────────────────────────────────────────────────

def cosine_lr(epoch: int, total: int, warmup: int, base: float,
              floor: float = 1e-6) -> float:
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * t))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    cfg  = CFG
    root = find_data_root(cfg["data_root"])

    print("─" * 60)
    print("FC Audio JEPA — Step 1: Supervised Baseline")
    print("─" * 60)

    # Species vocabulary
    species_list   = load_species_list(root)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes    = len(species_list)
    print(f"[MODEL] species: {num_classes}")

    ds_kwargs = dict(
        data_root        = root,
        species_to_idx   = species_to_idx,
        duration         = cfg["duration"],
        sample_rate      = cfg["sample_rate"],
        n_mels           = cfg["n_mels"],
        n_fft            = cfg["n_fft"],
        hop_length       = cfg["hop_length"],
        val_fraction     = cfg["val_fraction"],
        seed             = cfg["seed"],
    )

    # Context: soundscape windows (matches test distribution)
    sc_tr = SoundscapeDataset(split="train", **ds_kwargs)
    sc_va = SoundscapeDataset(split="val",   **ds_kwargs)

    # Target: curated species clips (strong species identity signal)
    cl_tr = BirdCLEF2026Dataset(
        split="train", secondary_weight=cfg["secondary_weight"],
        min_rating=cfg["min_rating"], **ds_kwargs
    )
    cl_va = BirdCLEF2026Dataset(
        split="val", secondary_weight=cfg["secondary_weight"],
        min_rating=cfg["min_rating"], **ds_kwargs
    )

    # Training: combine both; Validation: soundscape only (closest to test dist)
    train_ds = ConcatDataset([sc_tr, cl_tr]) if len(sc_tr) > 0 else cl_tr
    val_ds   = sc_va if len(sc_va) > 0 else cl_va
    print(f"[DATA]  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
    )

    # Spectrogram time dimension (consistent with model)
    n_frames = math.ceil(cfg["duration"] * cfg["sample_rate"] / cfg["hop_length"]) + 1
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] device={device}  spec=({cfg['n_mels']}, {n_frames})")

    # On CPU: reduce to a fast smoke-test (3 epochs, smaller batch)
    if device.type == "cpu":
        cfg = dict(cfg)
        cfg["epochs"]     = 3
        cfg["batch_size"] = 32
        cfg["num_workers"] = 2
        print("[TRAIN] CPU mode — epochs=3, batch_size=32 (smoke test)")

    model = BaselineClassifier(
        n_mels     = cfg["n_mels"],
        n_frames   = n_frames,
        patch_size = cfg["patch_size"],
        d          = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        depth      = cfg["n_layers"],
        dropout    = cfg["dropout"],
        n_cls      = num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] parameters: {n_params / 1e6:.2f}M")

    # Class-frequency-based positive weight from training labels (clips only, fast)
    cc = torch.zeros(num_classes)
    for sp in cl_tr.df["primary_label"].astype(str):
        if sp in species_to_idx:
            cc[species_to_idx[sp]] += 1
    pos_weight = ((len(cl_tr) - cc) / cc.clamp(min=1)).clamp(1.0, 50.0).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    ckpt_dir = "/kaggle/working/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_auc = 0.0

    for ep in range(cfg["epochs"]):
        lr = cosine_lr(ep, cfg["epochs"], cfg["warmup_epochs"], cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss, nb = 0.0, 0
        for specs, lbls in train_loader:
            specs, lbls = specs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(specs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            nb      += 1
        tr_loss /= max(nb, 1)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        pall, lall = [], []
        with torch.no_grad():
            for specs, lbls in val_loader:
                pall.append(torch.sigmoid(model(specs.to(device))).cpu().numpy())
                lall.append(lbls.numpy())
        pall = np.concatenate(pall)
        lall = np.concatenate(lall)
        # Hard labels for AUC: count primary labels only (> 0.9 threshold)
        lall_hard = (lall > 0.9).astype(float)
        valid     = lall_hard.sum(0) > 0
        val_auc   = (
            roc_auc_score(lall_hard[:, valid], pall[:, valid], average="macro")
            if valid.sum() > 0 else 0.0
        )

        print(
            f"Epoch {ep+1:02d}/{cfg['epochs']:02d} | "
            f"loss {tr_loss:.4f} | val_auc {val_auc:.4f} | lr {lr:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"epoch": ep + 1, "model_state": model.state_dict(),
                 "val_auc": val_auc, "cfg": cfg, "species": species_list},
                f"{ckpt_dir}/best.pt",
            )

    print(f"\nBest val AUC: {best_auc:.4f}  |  ckpt: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
