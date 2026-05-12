import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        data_root,
        taxonomy_csv,
        split="train",
        duration=5.0,
        sample_rate=32000,
        n_mels=128,
        n_fft=1024,
        hop_length=320,
        val_fraction=0.1,
        seed=42,
    ):
        self.data_root = data_root
        self.duration = duration
        self.sample_rate = sample_rate
        self.split = split
        self.target_samples = int(duration * sample_rate)

        taxonomy = pd.read_csv(taxonomy_csv)
        # BirdCLEF 2026 taxonomy uses 'species_code'; fall back to 'primary_label'
        sp_col = "species_code" if "species_code" in taxonomy.columns else "primary_label"
        self.species_list = sorted(taxonomy[sp_col].dropna().unique().tolist())
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        self.num_species = len(self.species_list)

        records = self._scan_audio(data_root)
        df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_n = max(1, int(len(df) * val_fraction))
        self.df = df.iloc[:val_n] if split == "val" else df.iloc[val_n:]
        self.df = self.df.reset_index(drop=True)

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=24)
        self.time_mask = T.TimeMasking(time_mask_param=80)
        self._resamplers = {}

    def _scan_audio(self, data_root):
        records = []
        audio_dir = os.path.join(data_root, "train_audio")
        if not os.path.isdir(audio_dir):
            return records
        for species in os.listdir(audio_dir):
            if species not in self.species_to_idx:
                continue
            sp_dir = os.path.join(audio_dir, species)
            if not os.path.isdir(sp_dir):
                continue
            for fname in os.listdir(sp_dir):
                if fname.lower().endswith((".ogg", ".mp3", ".wav", ".flac")):
                    records.append({"path": os.path.join(sp_dir, fname), "species": species})
        return records

    def __len__(self):
        return len(self.df)

    def _load_wave(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav  # (1, N)

    def _crop_or_pad(self, wav):
        n = wav.shape[-1]
        if n >= self.target_samples:
            if self.split == "train":
                start = torch.randint(0, n - self.target_samples + 1, (1,)).item()
            else:
                start = (n - self.target_samples) // 2
            return wav[..., start : start + self.target_samples]
        return torch.nn.functional.pad(wav, (0, self.target_samples - n))

    def _make_label(self, species):
        label = torch.zeros(self.num_species)
        if species in self.species_to_idx:
            label[self.species_to_idx[species]] = 1.0
        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = self._load_wave(row["path"])
        wav = self._crop_or_pad(wav)

        spec = self.mel(wav)           # (1, n_mels, T)
        spec = self.to_db(spec)        # log scale
        spec = (spec + 80.0) / 80.0   # normalize to ~[0, 1]
        spec = spec.clamp(0.0, 1.0)

        if self.split == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        label = self._make_label(row["species"])
        return spec, label
