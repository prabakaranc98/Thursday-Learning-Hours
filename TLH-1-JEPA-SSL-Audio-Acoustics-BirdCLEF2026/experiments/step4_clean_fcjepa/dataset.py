"""
Step 4 Clean FC-JEPA — Dataset

BirdCLEFStep4Dataset:
  Context input   : real soundscape window (E_c input)
  Target inputs   : label-conditioned clean audio per present species (E_t inputs)

  Empty window  → [white_noise(5s)]          n_active=1
  Labeled window → [stitch(s1), stitch(s2), ...]  n_active=|S|, capped at K_MAX

Returns per __getitem__:
    spec_ctx     : (1, n_mels, n_frames)          soundscape window
    spec_targets : list of (1, n_mels, n_frames)  per-source clean specs, len=n_active
    y_label      : (n_cls,)                        multilabel classification target
    n_active     : int                             number of active sources
"""

import ast
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


def _parse_hms(t):
    parts = str(t).strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600.0 + m * 60.0 + s


class BirdCLEFStep4Dataset(Dataset):
    K_MAX = 6  # max sources per window (slots cap)

    def __init__(
        self,
        data_root,
        species_to_idx,
        split="train",
        duration=5.0,
        sample_rate=32_000,
        n_mels=128,
        n_fft=1024,
        hop_length=320,
        val_fraction=0.15,
        seed=42,
        include_empty_windows=True,
        **_,
    ):
        self.species_to_idx = species_to_idx
        self.num_classes    = len(species_to_idx)
        self.split          = split
        self.sample_rate    = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.n_mels         = n_mels

        self.sc_dir    = os.path.join(data_root, "train_soundscapes")
        self.audio_dir = os.path.join(data_root, "train_audio")

        self.mel   = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
        self.fmask = T.FrequencyMasking(freq_mask_param=24) if split == "train" else None
        self.tmask = T.TimeMasking(time_mask_param=80)     if split == "train" else None
        self._resamplers: dict = {}

        # Per-species clip index
        self.species_clips: dict[str, list[str]] = defaultdict(list)
        if os.path.isdir(self.audio_dir):
            for sp in os.listdir(self.audio_dir):
                sp_dir = os.path.join(self.audio_dir, sp)
                if not (os.path.isdir(sp_dir) and sp in species_to_idx):
                    continue
                clips = sorted(
                    os.path.join(sp_dir, f)
                    for f in os.listdir(sp_dir)
                    if f.lower().endswith((".ogg", ".mp3", ".flac", ".wav"))
                )
                if clips:
                    self.species_clips[sp] = clips
        print(f"[DATA] Step4 {split}: {len(self.species_clips)} species with clips")

        # Soundscape windows
        lbl_csv = os.path.join(data_root, "train_soundscapes_labels.csv")
        if not os.path.isfile(lbl_csv) or not os.path.isdir(self.sc_dir):
            print("[DATA] BirdCLEFStep4Dataset: soundscape labels missing — 0 samples")
            self.windows: list = []
            return

        df = pd.read_csv(lbl_csv)
        df = df.drop_duplicates(["filename", "start"]).reset_index(drop=True)
        df["start_s"] = df["start"].apply(_parse_hms)

        def _parts(x):
            return [s.strip() for s in str(x).split(";") if s.strip()] if pd.notna(x) else []

        # species_set_all: all annotated species in taxonomy → used for y label (AUC target)
        # species_set_clips: subset that also have curated clips → used for JEPA targets (E_t)
        df["species_set_all"]   = df["primary_label"].apply(
            lambda x: frozenset(p for p in _parts(x) if p in species_to_idx))
        df["species_set_clips"] = df["primary_label"].apply(
            lambda x: frozenset(p for p in _parts(x)
                                 if p in species_to_idx and p in self.species_clips))

        # Train/val split by file
        files = sorted(df["filename"].unique())
        rng   = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val     = max(1, int(len(files) * val_fraction))
        val_files = set(files[:n_val])
        mask = df["filename"].isin(val_files)
        df   = (df[mask] if split == "val" else df[~mask]).reset_index(drop=True)

        if not include_empty_windows:
            df = df[df["species_set_all"].apply(len) > 0].reset_index(drop=True)

        self.windows = []
        for _, row in df.iterrows():
            sp_all   = row["species_set_all"]
            sp_clips = row["species_set_clips"]

            # Clamp JEPA targets to K_MAX species (those with clips, alphabetically)
            if len(sp_clips) > self.K_MAX:
                sp_clips = frozenset(sorted(sp_clips)[: self.K_MAX])

            # y label: 1.0 for ALL annotated species (confirmed expert annotations)
            y = torch.zeros(self.num_classes)
            for sp in sp_all:
                y[species_to_idx[sp]] = 1.0

            self.windows.append(
                (row["filename"], float(row["start_s"]), sp_clips, y)
            )

        n_empty   = sum(1 for _, _, s, _ in self.windows if len(s) == 0)
        n_partial = sum(1 for _, _, s, _ in self.windows if 0 < len(s) < self.K_MAX)
        print(
            f"[DATA] Step4 {split}: {len(self.windows)} windows "
            f"({df['filename'].nunique()} files, {n_empty} empty, "
            f"{n_partial} partial-clip)"
        )

    def __len__(self):
        return len(self.windows)

    # ── Audio helpers ───────────────────────────────────────────────────────────

    def _resample(self, wav, sr):
        if sr == self.sample_rate:
            return wav
        if sr not in self._resamplers:
            self._resamplers[sr] = T.Resample(sr, self.sample_rate)
        return self._resamplers[sr](wav)

    def _stitch_clips(self, clips: list[str]) -> torch.Tensor:
        """Concatenate random clips from a species to fill target_samples."""
        buf, filled = [], 0
        clips_shuffled = clips.copy()
        random.shuffle(clips_shuffled)
        ci = 0
        while filled < self.target_samples:
            path = clips_shuffled[ci % len(clips_shuffled)]
            ci  += 1
            try:
                wav, sr = torchaudio.load(path)
            except Exception:
                if ci > len(clips_shuffled) * 3:
                    break
                continue
            wav = self._resample(wav, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            clip_len = wav.shape[-1]
            need     = self.target_samples - filled
            if clip_len > need:
                start = random.randint(0, clip_len - need)
                wav   = wav[..., start : start + need]
            buf.append(wav)
            filled += wav.shape[-1]
        if not buf:
            return torch.zeros(1, self.target_samples)
        return torch.cat(buf, dim=-1)[..., : self.target_samples]

    def _to_spec(self, wav: torch.Tensor, augment: bool = False) -> torch.Tensor:
        spec = self.to_db(self.mel(wav))
        spec = ((spec + 80.0) / 80.0).clamp(0.0, 1.0)
        if augment and self.fmask is not None:
            spec = self.fmask(spec)
            spec = self.tmask(spec)
        return spec

    # ── __getitem__ ─────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        filename, start_s, species_set, y = self.windows[idx]
        is_train = self.split == "train"

        # Context: soundscape window (seek + read only the needed frames)
        path = os.path.join(self.sc_dir, filename)
        try:
            start_frame = int(start_s * self.sample_rate)
            wav_ctx, sr = torchaudio.load(
                path,
                frame_offset=start_frame,
                num_frames=self.target_samples,
            )
            wav_ctx = self._resample(wav_ctx, sr)
            if wav_ctx.shape[0] > 1:
                wav_ctx = wav_ctx.mean(0, keepdim=True)
            if wav_ctx.shape[-1] < self.target_samples:
                wav_ctx = F.pad(wav_ctx, (0, self.target_samples - wav_ctx.shape[-1]))
        except Exception as e:
            print(f"[WARN] ctx {path}: {e}")
            wav_ctx = torch.zeros(1, self.target_samples)

        spec_ctx = self._to_spec(wav_ctx, augment=is_train)

        # Targets: label-conditioned clean audio
        spec_targets: list[torch.Tensor] = []
        if len(species_set) == 0:
            # Empty window → quiet white noise as background target
            wav_bg = torch.randn(1, self.target_samples) * 0.02
            spec_targets.append(self._to_spec(wav_bg))
        else:
            for sp in sorted(species_set):
                clips = self.species_clips.get(sp, [])
                if clips:
                    wav_sp = self._stitch_clips(clips)
                else:
                    wav_sp = torch.zeros(1, self.target_samples)
                spec_targets.append(self._to_spec(wav_sp))

        return spec_ctx, spec_targets, y, len(spec_targets)


class ClipStep4Dataset(Dataset):
    """
    Curated train_audio clips as single-species JEPA examples.

    Each item treats one clip as a soundscape window (n_active=1).
    Context  = the clip itself (E_c input).
    Target   = a DIFFERENT random crop/stitch from the same species (E_t input).

    This avoids Mode 1 (BYOL) by using different crops for context and target,
    adding ~200K training samples to supplement the small soundscape labeled set.
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
        val_fraction: float = 0.15,
        seed: int = 42,
        **_,
    ):
        self.species_to_idx = species_to_idx
        self.num_classes    = len(species_to_idx)
        self.split          = split
        self.sample_rate    = sample_rate
        self.target_samples = int(duration * sample_rate)

        self.mel   = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        )
        self.to_db  = T.AmplitudeToDB(top_db=80)
        self.fmask  = T.FrequencyMasking(freq_mask_param=24) if split == "train" else None
        self.tmask  = T.TimeMasking(time_mask_param=80)     if split == "train" else None
        self._resamplers: dict = {}

        # Load all clips per species from train.csv
        audio_dir = os.path.join(data_root, "train_audio")
        train_csv = os.path.join(data_root, "train.csv")
        if not os.path.isfile(train_csv) or not os.path.isdir(audio_dir):
            print(f"[DATA] ClipStep4Dataset: missing train.csv or train_audio/ — 0 samples")
            self.items: list = []
            return

        df = pd.read_csv(train_csv)
        df["primary_label"] = df["primary_label"].astype(str)
        df = df[df["primary_label"].isin(species_to_idx)].copy()
        df["filepath"] = df["filename"].apply(lambda f: os.path.join(audio_dir, f))
        df = df[df["filepath"].apply(os.path.isfile)].reset_index(drop=True)

        # Parse secondary_labels → weak supervision for co-occurring species
        def _parse_secondary(raw):
            try:
                items = ast.literal_eval(str(raw))
                return [str(s).strip() for s in items if str(s).strip() in species_to_idx]
            except Exception:
                return []
        df["secondary_parsed"] = df.get("secondary_labels", pd.Series(["[]"] * len(df))).apply(
            _parse_secondary)

        # Train/val split
        rng = np.random.default_rng(seed)
        df  = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        cut = max(1, int(len(df) * val_fraction))
        df  = (df.iloc[:cut] if split == "val" else df.iloc[cut:]).reset_index(drop=True)

        # Per-species clip index for target sampling
        self.by_species: dict[str, list[str]] = defaultdict(list)
        for _, row in df.iterrows():
            self.by_species[row["primary_label"]].append(row["filepath"])

        # Items: (filepath, primary_sp, secondary_sps)
        self.items = [
            (row["filepath"], row["primary_label"], row["secondary_parsed"])
            for _, row in df.iterrows()
        ]
        print(f"[DATA] ClipStep4 {split}: {len(self.items)} clips "
              f"({df['primary_label'].nunique()} species)")

    def __len__(self):
        return len(self.items)

    def _load_wave(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n >= self.target_samples:
            start = random.randint(0, n - self.target_samples) if self.split == "train" else 0
            return wav[..., start : start + self.target_samples]
        return F.pad(wav, (0, self.target_samples - n))

    def _to_spec(self, wav: torch.Tensor, augment: bool = False) -> torch.Tensor:
        spec = self.to_db(self.mel(wav))
        spec = ((spec + 80.0) / 80.0).clamp(0.0, 1.0)
        if augment and self.fmask is not None:
            spec = self.fmask(spec)
            spec = self.tmask(spec)
        return spec

    def __getitem__(self, idx):
        ctx_path, sp, secondary_sps = self.items[idx]
        is_train = self.split == "train"

        # Context: one crop from this clip
        try:
            wav_ctx = self._crop(self._load_wave(ctx_path))
        except Exception:
            wav_ctx = torch.zeros(1, self.target_samples)
        spec_ctx = self._to_spec(wav_ctx, augment=is_train)

        # Target: a DIFFERENT crop from the same species (avoids BYOL identical-view)
        all_clips = self.by_species.get(sp, [ctx_path])
        tgt_path  = random.choice(all_clips)
        try:
            wav_tgt = self._crop(self._load_wave(tgt_path))
        except Exception:
            wav_tgt = torch.zeros(1, self.target_samples)
        spec_tgt = self._to_spec(wav_tgt)

        # Label: 1.0 for primary, 0.5 for secondary (weak label for co-occurrences)
        y = torch.zeros(self.num_classes)
        y[self.species_to_idx[sp]] = 1.0
        for s in secondary_sps:
            if s in self.species_to_idx:
                y[self.species_to_idx[s]] = max(y[self.species_to_idx[s]].item(), 0.5)

        # Single-species JEPA: n_active=1, one target
        return spec_ctx, [spec_tgt], y, 1


def step4_collate_fn(batch):
    """Collate variable-length target lists into fixed K_MAX tensors.
    Handles n_active=0 (unlabeled soundscapes) with zero-padded targets."""
    K_MAX  = BirdCLEFStep4Dataset.K_MAX
    _dummy = None  # lazy-init shape reference

    spec_ctxs   = []
    all_targets = [[] for _ in range(K_MAX)]
    ys          = []
    n_actives   = []

    for spec_ctx, spec_targets, y, n_active in batch:
        spec_ctxs.append(spec_ctx)
        ys.append(y)
        n_actives.append(n_active)
        if _dummy is None:
            _dummy = torch.zeros_like(spec_ctx)  # (1, n_mels, n_frames)
        pad_spec = _dummy
        for k in range(K_MAX):
            all_targets[k].append(spec_targets[k] if k < len(spec_targets) else pad_spec)

    return (
        torch.stack(spec_ctxs),                   # (B, 1, n_mels, n_frames)
        [torch.stack(t) for t in all_targets],    # list of K_MAX × (B, 1, n_mels, n_frames)
        torch.stack(ys),                           # (B, n_cls)
        torch.tensor(n_actives, dtype=torch.long), # (B,)
    )


class UnlabeledSoundscapeDataset(Dataset):
    """
    ALL train_soundscapes (labeled + unlabeled) as self-supervised context windows.

    For each file, generates non-overlapping 5-second windows. Returns n_active=0
    so the training loop applies only SIGReg + EMA update (no JEPA target needed).

    This exposes the encoder to ~127K real Pantanal soundscape windows, the same
    acoustic environment as the hidden test set.

    n_per_epoch: if set, randomly subsample this many windows per epoch to keep
    training speed manageable when combined with labeled datasets.
    """

    def __init__(
        self,
        data_root: str,
        num_classes: int,
        duration: float = 5.0,
        sample_rate: int = 32_000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_per_epoch: int = 8192,
        seed: int = 42,
        **_,
    ):
        self.num_classes    = num_classes
        self.sample_rate    = sample_rate
        self.target_samples = int(duration * sample_rate)
        self.n_per_epoch    = n_per_epoch

        self.mel   = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
        self._resamplers: dict = {}

        sc_dir = os.path.join(data_root, "train_soundscapes")
        if not os.path.isdir(sc_dir):
            print("[DATA] UnlabeledSoundscape: train_soundscapes/ not found — 0 windows")
            self._windows: list = []
            self._epoch_idxs: list = []
            return

        windows_per_file = 60 // int(duration)   # 1-min files → 12 windows each
        all_files = sorted(
            os.path.join(sc_dir, f)
            for f in os.listdir(sc_dir)
            if f.lower().endswith(".ogg")
        )
        self._windows = [
            (fpath, w * int(duration))
            for fpath in all_files
            for w in range(windows_per_file)
        ]
        print(f"[DATA] UnlabeledSoundscape: {len(all_files)} files → "
              f"{len(self._windows)} windows (subsampled to {n_per_epoch}/epoch)")

        rng = np.random.default_rng(seed)
        self._epoch_idxs = rng.choice(
            len(self._windows), size=min(n_per_epoch, len(self._windows)), replace=False
        ).tolist()

    def reshuffle(self, seed: int) -> None:
        """Call at start of each epoch to get a fresh subsample."""
        rng = np.random.default_rng(seed)
        self._epoch_idxs = rng.choice(
            len(self._windows), size=min(self.n_per_epoch, len(self._windows)), replace=False
        ).tolist()

    def __len__(self):
        return len(self._epoch_idxs)

    def __getitem__(self, idx):
        fpath, start_s = self._windows[self._epoch_idxs[idx]]
        try:
            start_frame = start_s * self.sample_rate
            wav, sr = torchaudio.load(fpath, frame_offset=start_frame,
                                      num_frames=self.target_samples)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != self.sample_rate:
                if sr not in self._resamplers:
                    self._resamplers[sr] = T.Resample(sr, self.sample_rate)
                wav = self._resamplers[sr](wav)
            if wav.shape[-1] < self.target_samples:
                wav = F.pad(wav, (0, self.target_samples - wav.shape[-1]))
        except Exception:
            wav = torch.zeros(1, self.target_samples)

        spec = self.to_db(self.mel(wav))
        spec = ((spec + 80.0) / 80.0).clamp(0.0, 1.0)

        # n_active=0 signals "no JEPA target" to the training loop
        return spec, [], torch.zeros(self.num_classes), 0
