#!/usr/bin/env python3
"""
FC-Audio-JEPA: Forward Pass Demonstration
==========================================
Uses REAL BirdCLEF 2026 audio — one labeled soundscape window containing
two frog species, plus one curated clip per species as the target.

Demo window:
  Soundscape : BC2026_Train_0019_S22_20211104_200000.ogg  [0s, 5s]
  Species A  : 22973  Leptodactylus fuscus  (Whistling Grass Frog)
  Species B  : 65380  Dendropsophus nanus   (Dwarf Tree Frog)
  Clip A     : 22973/iNat28425.ogg
  Clip B     : 65380/iNat1650638.ogg

Run:
  /opt/anaconda3/bin/python3 forward_pass_demo.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# ── paths (edit if you store audio elsewhere) ─────────────────────────────────
SOUNDSCAPE = "/tmp/birdclef_sample/audio/soundscapes/BC2026_Train_0019_S22_20211104_200000.ogg"
CLIP_A     = "/tmp/birdclef_sample/audio/clips/iNat28425.ogg"      # Whistling Grass Frog
CLIP_B     = "/tmp/birdclef_sample/audio/clips/iNat1650638.ogg"    # Dwarf Tree Frog

# ── config ────────────────────────────────────────────────────────────────────
SR         = 32_000
DURATION   = 5.0          # seconds per window
N_MELS     = 64
N_FFT      = 1024
HOP        = 320
PATCH_H    = 16            # frequency patches
PATCH_W    = 16            # time patches
D          = 128           # embedding dimension
K          = 4             # number of slots
N_HEADS    = 4
N_SPECIES  = 234           # full BirdCLEF label space

torch.manual_seed(0)       # reproducible random weights
SEP = "─" * 60


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class LogMel(nn.Module):
    """Waveform → log-mel spectrogram."""
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
        self.db  = T.AmplitudeToDB(top_db=80)

    def forward(self, wav):
        # wav: (1, samples)
        return self.db(self.mel(wav))   # (1, N_MELS, T)


class PatchEmbed(nn.Module):
    """
    Slice log-mel spectrogram into non-overlapping patches, project each to D.

    Input  : (B, 1, F, T)    log-mel spectrogram
    Output : (B, N_patches, D)

    Each patch covers PATCH_H mel bins × PATCH_W time frames.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, D, kernel_size=(PATCH_H, PATCH_W),
                              stride=(PATCH_H, PATCH_W))

    def forward(self, x):
        x = self.proj(x)            # (B, D, F//ph, T//pw)
        B, D_, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.attn  = nn.MultiheadAttention(D, N_HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(D)
        self.ff    = nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """
    f_θ : spectrogram → patch token sequence

    Input  : (B, 1, F, T)
    Output : (B, N_patches, D)
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.blocks      = nn.Sequential(TransformerBlock(), TransformerBlock())
        self.norm        = nn.LayerNorm(D)

    def forward(self, x):
        tokens = self.patch_embed(x)    # (B, N, D)
        tokens = self.blocks(tokens)    # (B, N, D)
        return self.norm(tokens)        # (B, N, D)


class SlotAttention(nn.Module):
    """
    P1 : patch tokens → K factorised slot vectors

    Each slot learns to attend to a different part of the spectrogram,
    corresponding to one acoustic source (species or background).

    Slots compete for patches via softmax over the slot dimension:
        attn[k, n]  =  how much slot k "owns" patch n
    Attention is normalised over patches so slots can't all attend to everything.

    Input  : (B, N_patches, D)
    Output : slots (B, K, D),  attn_map (B, K, N_patches)
    """
    def __init__(self):
        super().__init__()
        self.slots_mu     = nn.Parameter(torch.randn(1, K, D))
        self.slots_logsig = nn.Parameter(torch.zeros(1, K, D))
        self.norm_in      = nn.LayerNorm(D)
        self.norm_slots   = nn.LayerNorm(D)
        self.norm_ff      = nn.LayerNorm(D)
        self.to_q         = nn.Linear(D, D, bias=False)
        self.to_k         = nn.Linear(D, D, bias=False)
        self.to_v         = nn.Linear(D, D, bias=False)
        self.gru          = nn.GRUCell(D, D)
        self.ff           = nn.Sequential(nn.Linear(D, D*2), nn.ReLU(),
                                          nn.Linear(D*2, D))
        self.scale        = D ** -0.5

    def forward(self, x, n_iters=3):
        B, N, _ = x.shape
        x_n = self.norm_in(x)
        k   = self.to_k(x_n)       # (B, N, D)
        v   = self.to_v(x_n)       # (B, N, D)

        # initialise slots from learnable Gaussian
        slots = self.slots_mu.expand(B, -1, -1)
        slots = slots + self.slots_logsig.exp() * torch.randn_like(slots)

        for _ in range(n_iters):
            prev   = slots
            q      = self.to_q(self.norm_slots(slots))          # (B, K, D)
            dots   = torch.einsum('bkd,bnd->bkn', q, k) * self.scale   # (B, K, N)

            # softmax over K → each patch assigns attention mass across slots
            attn   = dots.softmax(dim=1)                        # (B, K, N)
            # normalise over N → weighted mean per slot
            attn_n = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)

            updates = torch.einsum('bkn,bnd->bkd', attn_n, v)  # (B, K, D)

            slots = self.gru(
                updates.reshape(B * K, D),
                prev.reshape(B * K, D),
            ).reshape(B, K, D)
            slots = slots + self.ff(self.norm_ff(slots))

        return slots, attn  # (B,K,D), (B,K,N)


class ComposePredictor(nn.Module):
    """
    P2 : K slots → predicted composition ẑ_comp

    Takes all K slot vectors (the factorised representation) and predicts
    what the target encoder would output for the mixture of species present.

    Input  : (B, K, D)
    Output : (B, D)   — NOT normalised; MSE loss will drive it toward unit norm
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.Linear(K, 1)         # learned slot aggregation
        self.mlp  = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, slots):
        # slots: (B, K, D)
        pooled = self.pool(slots.transpose(1, 2)).squeeze(-1)   # (B, D)
        return self.mlp(pooled)                                  # (B, D)


class Classifier(nn.Module):
    """
    Clf : K slots → N_SPECIES logits  (multi-label, sigmoid applied at loss)

    Reads all K slots and predicts which species are present in the window.
    Input  : (B, K, D)
    Output : (B, N_SPECIES)
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(K * D, N_SPECIES)

    def forward(self, slots):
        return self.head(slots.reshape(slots.size(0), -1))   # (B, N_SPECIES)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_ogg(path):
    """Load ogg via soundfile (works without torchcodec)."""
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(path, always_2d=True)   # (samples, channels)
    wav = torch.from_numpy(data.T.astype(np.float32))  # (channels, samples)
    return wav, sr


def load_window(path, start_sec=0.0, duration=DURATION):
    """Load a fixed-length window from an audio file, resample to SR."""
    wav, sr = _load_ogg(path)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav)
    wav = wav.mean(0, keepdim=True)         # mono
    s   = int(start_sec * SR)
    e   = s + int(duration * SR)
    if e > wav.shape[1]:
        wav = F.pad(wav, (0, e - wav.shape[1]))
    return wav[:, s:e]                      # (1, samples)


def load_clip(path, duration=DURATION):
    """Load a species clip, take a center crop of `duration` seconds."""
    wav, sr = _load_ogg(path)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav)
    wav = wav.mean(0, keepdim=True)
    n   = int(duration * SR)
    if wav.shape[1] >= n:
        start = (wav.shape[1] - n) // 2
        wav   = wav[:, start:start + n]
    else:
        wav = F.pad(wav, (0, n - wav.shape[1]))
    return wav                              # (1, samples)


# ══════════════════════════════════════════════════════════════════════════════
# HUNGARIAN MATCHING  (scipy-free, O(K*N) for small K)
# ══════════════════════════════════════════════════════════════════════════════

def hungarian_mse_cost(slots_n, z_each):
    """
    slots_n : (K, D)  normalised slot vectors
    z_each  : (N, D)  normalised target vectors (N ≤ K)
    Returns  : list of (slot_idx, target_idx) pairs — optimal assignment
    """
    from scipy.optimize import linear_sum_assignment
    cost = torch.cdist(slots_n, z_each, p=2).detach().cpu().numpy()  # (K, N)
    row, col = linear_sum_assignment(cost[:z_each.shape[0]].T)       # N rows
    return list(zip(col.tolist(), row.tolist()))   # (slot_idx, species_idx)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════════════════════════════

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def shape_stats(name, t):
    mn, mx, std = t.min().item(), t.max().item(), t.std().item()
    print(f"  {name:30s}  shape={tuple(t.shape)}  "
          f"min={mn:+.3f}  max={mx:+.3f}  std={std:.3f}")


def main():
    print(f"\n{'═'*60}")
    print("  Factorize-and-Compose Audio-JEPA — Forward Pass Demo")
    print(f"{'═'*60}")
    print(f"  Soundscape window: BC2026_Train_0019, t=[0s, 5s]")
    print(f"  Species A (22973): Leptodactylus fuscus  — Whistling Grass Frog")
    print(f"  Species B (65380): Dendropsophus nanus   — Dwarf Tree Frog")

    # ── instantiate models ────────────────────────────────────────────────────
    log_mel    = LogMel()
    f_theta    = ContextEncoder()          # context encoder  (trainable)
    f_tbar     = ContextEncoder()          # target encoder   (EMA copy, no grad)
    f_tbar.load_state_dict(f_theta.state_dict())
    P1         = SlotAttention()
    P2         = ComposePredictor()
    Clf        = Classifier()

    n_params = sum(p.numel() for p in
                   list(f_theta.parameters()) + list(P1.parameters()) +
                   list(P2.parameters()) + list(Clf.parameters()))
    print(f"\n  Trainable parameters: {n_params / 1e6:.2f}M")
    print(f"  Slots K={K},  d={D},  patch={PATCH_H}×{PATCH_W},  mels={N_MELS}")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 0 — Load audio from disk")
    # ══════════════════════════════════════════════════════════════════════════

    wav_w  = load_window(SOUNDSCAPE, start_sec=0.0)   # soundscape window
    wav_a  = load_clip(CLIP_A)                         # species A clip
    wav_b  = load_clip(CLIP_B)                         # species B clip

    print(f"  Soundscape window    {tuple(wav_w.shape)}  "
          f"({wav_w.shape[1]/SR:.1f}s @ {SR}Hz)")
    print(f"  Species A clip       {tuple(wav_a.shape)}  "
          f"({wav_a.shape[1]/SR:.1f}s @ {SR}Hz)")
    print(f"  Species B clip       {tuple(wav_b.shape)}  "
          f"({wav_b.shape[1]/SR:.1f}s @ {SR}Hz)")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 1 — Log-mel spectrograms")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   wav (1, samples) → LogMel → S (1, N_MELS, T)
    #
    #   Axis meanings:
    #     dim 0 → channel (always 1 — mono)
    #     dim 1 → frequency bins (mel scale, low → high)
    #     dim 2 → time frames (left → right)
    #

    S_w = log_mel(wav_w.unsqueeze(0))   # (B=1, 1, F, T)
    S_a = log_mel(wav_a.unsqueeze(0))
    S_b = log_mel(wav_b.unsqueeze(0))

    # pad time to be divisible by PATCH_W
    def pad_time(S):
        T = S.shape[-1]
        pad = (PATCH_W - T % PATCH_W) % PATCH_W
        return F.pad(S, (0, pad))

    S_w = pad_time(S_w)
    S_a = pad_time(S_a)
    S_b = pad_time(S_b)

    F_dim = S_w.shape[2]
    T_dim = S_w.shape[3]
    N_patches = (F_dim // PATCH_H) * (T_dim // PATCH_W)

    print(f"  Spectrogram shape:  (B=1, C=1, F={F_dim}, T={T_dim})")
    print(f"  Patch grid:         {F_dim//PATCH_H} freq × {T_dim//PATCH_W} time"
          f"  = {N_patches} patches total")
    shape_stats("S_w  soundscape mel", S_w.squeeze())
    shape_stats("S_a  species A mel",  S_a.squeeze())
    shape_stats("S_b  species B mel",  S_b.squeeze())

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 2 — Target path  (stop-gradient, f_θ̄)")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   Each species clip → f_θ̄ → mean-pool patches → L2-normalize → z_i
    #
    #   z_i  lives on the unit hypersphere in R^D.
    #   The composition z_comp = normalize(z_A + z_B) is the regression target
    #   for this soundscape window.
    #
    #   stop_gradient: NO gradients flow through f_θ̄ or z_i.
    #

    with torch.no_grad():
        # encode each clip: mean-pool the patch token sequence → (1, D)
        tokens_a = f_tbar(S_a)          # (1, N_patches, D)
        tokens_b = f_tbar(S_b)

        z_a_raw = tokens_a.mean(dim=1)  # (1, D)  — pool over patches
        z_b_raw = tokens_b.mean(dim=1)

        z_a = F.normalize(z_a_raw, dim=-1)   # unit norm  → z_A
        z_b = F.normalize(z_b_raw, dim=-1)   # unit norm  → z_B

        # latent composition — NOT audio mixing
        z_comp = F.normalize(z_a + z_b, dim=-1)   # (1, D)

    shape_stats("z_A  (species A latent)", z_a.squeeze())
    shape_stats("z_B  (species B latent)", z_b.squeeze())
    shape_stats("z_comp = norm(z_A + z_B)", z_comp.squeeze())

    cos_ab   = (z_a * z_b).sum().item()
    cos_a_c  = (z_a * z_comp).sum().item()
    cos_b_c  = (z_b * z_comp).sum().item()
    print(f"\n  Cosine similarities between targets:")
    print(f"    cos(z_A, z_B)    = {cos_ab:+.4f}  "
          f"({'similar' if cos_ab > 0.7 else 'distinct'})")
    print(f"    cos(z_A, z_comp) = {cos_a_c:+.4f}  (z_comp should be between A and B)")
    print(f"    cos(z_B, z_comp) = {cos_b_c:+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 3 — Context path  (f_θ, gradients flow)")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   soundscape window → f_θ → patch tokens h
    #
    #   h contains one embedding per spectrogram patch.
    #   These are the "observations" — f_θ encodes the noisy mixture.
    #

    h = f_theta(S_w)    # (1, N_patches, D)
    shape_stats("h = f_θ(soundscape)", h.squeeze())

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 4 — P1: Slot Attention  (factorize)")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   h → P1 → K slots  +  attention map
    #
    #   Slot k attends over all N patches; the softmax over K ensures
    #   each patch is "owned" by one dominant slot.
    #
    #   At convergence, slot k ≈ z_{π(k)}  for the matched species π(k).
    #   Unmatched slots (K > N labeled species) capture background.
    #

    slots, attn_map = P1(h)     # (1, K, D),  (1, K, N_patches)
    shape_stats("slots  (B, K, D)", slots.squeeze())
    shape_stats("attn_map  (K, N_patches)", attn_map.squeeze())

    # show attention entropy per slot — high entropy = attending broadly (background)
    eps  = 1e-8
    attn_sq = attn_map.squeeze()                              # (K, N)
    attn_sq = attn_sq / (attn_sq.sum(dim=1, keepdim=True) + eps)
    entropy = -(attn_sq * (attn_sq + eps).log()).sum(dim=1)   # (K,)
    max_H   = math.log(N_patches)
    print(f"\n  Attention entropy per slot  (max possible = {max_H:.2f}):")
    for k in range(K):
        bar  = "█" * int(20 * entropy[k].item() / max_H)
        note = "← broad (background?)" if entropy[k] > 0.7 * max_H else ""
        print(f"    slot {k}:  H = {entropy[k]:.2f}  {bar}  {note}")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 5 — P2: Compose Predictor")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   slots → P2 → ẑ_comp
    #
    #   P2 takes all K slot vectors and predicts what the composed target
    #   latent should be — i.e. what f_θ̄ would produce for the species mixture.
    #
    #   ẑ_comp is NOT normalised — MSE loss drives it toward the unit sphere.
    #

    z_hat_comp = P2(slots)      # (1, D)
    shape_stats("ẑ_comp = P2(slots)", z_hat_comp.squeeze())
    print(f"  ‖ẑ_comp‖ = {z_hat_comp.norm().item():.4f}  "
          f"(target ‖z_comp‖ = {z_comp.norm().item():.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 6 — Classifier head")
    # ══════════════════════════════════════════════════════════════════════════
    #
    #   slots → Clf → logits  (234 species, sigmoid for multi-label)
    #

    logits = Clf(slots)         # (1, 234)
    probs  = torch.sigmoid(logits)
    shape_stats("logits  (1, N_SPECIES)", logits.squeeze())
    print(f"  Predicted prob range: [{probs.min():.3f}, {probs.max():.3f}]")
    top5_idx  = probs.squeeze().topk(5).indices.tolist()
    top5_prob = probs.squeeze().topk(5).values.tolist()
    print(f"  Top-5 predicted species (random init — should be ~0.5):")
    for idx, prob in zip(top5_idx, top5_prob):
        print(f"    species index {idx:>3d}  →  prob = {prob:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 7 — Losses")
    # ══════════════════════════════════════════════════════════════════════════

    # ── L_comp: MSE between predicted and target composition ──────────────────
    #
    #   ẑ_comp  vs  z_comp  (unit norm)
    #   Both represent "the mixture of species A and B in latent space".
    #   At convergence: ‖ẑ_comp - z_comp‖ → 0
    #
    L_comp = F.mse_loss(z_hat_comp, z_comp)

    # ── L_fact: MSE per slot, after Hungarian matching ────────────────────────
    #
    #   z_each: (N, D) = [z_A, z_B]  — the individual species targets
    #   slots:  (K, D)  K=4 slots, N=2 targets → 2 slots matched, 2 unmatched
    #
    #   Matching: normalise slots, find assignment with minimum L2 cost,
    #   then compute MSE on the RAW (un-normalised) matched slots.
    #
    z_each   = torch.cat([z_a, z_b], dim=0)                    # (N=2, D)
    slots_sq = slots.squeeze(0)                                  # (K, D)
    slots_n  = F.normalize(slots_sq.detach(), dim=-1)           # (K, D) for assignment
    pairs    = hungarian_mse_cost(slots_n, z_each)              # [(slot_k, species_i), ...]

    print(f"  Hungarian assignment  (K={K} slots → N=2 species targets):")
    for slot_k, sp_i in pairs:
        sp_name = ["22973 Whistling Grass Frog", "65380 Dwarf Tree Frog"][sp_i]
        d_cos   = 1 - (F.normalize(slots_sq[slot_k], dim=-1) * z_each[sp_i]).sum().item()
        print(f"    slot {slot_k}  →  species {sp_name}   1-cos = {d_cos:.4f}")

    unmatched = [k for k in range(K) if k not in [p[0] for p in pairs]]
    print(f"  Unmatched slots: {unmatched}  ← learn background / silence")

    fact_terms = [F.mse_loss(slots_sq[k], z_each[i]) for k, i in pairs]
    L_fact = sum(fact_terms) / len(fact_terms)

    # ── L_cls: BCE for species presence ──────────────────────────────────────
    #
    #   True labels: species 22973 and 65380 are present → index into species list.
    #   Here we use placeholder indices 0 and 1 since we don't have the full
    #   taxonomy mapping at demo time.
    #
    multihot = torch.zeros(1, N_SPECIES)
    multihot[0, 0] = 1.0   # placeholder for species 22973
    multihot[0, 1] = 1.0   # placeholder for species 65380
    L_cls = F.binary_cross_entropy_with_logits(logits, multihot)

    λ1, λ2 = 0.5, 0.5
    L_total = L_cls + λ1 * L_fact + λ2 * L_comp

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  L_cls   (BCE species labels)  {L_cls.item():>6.4f} │")
    print(f"  │  L_fact  (slot → species MSE)  {L_fact.item():>6.4f} │")
    print(f"  │  L_comp  (ẑ_comp → z_comp MSE) {L_comp.item():>6.4f} │")
    print(f"  │  L_total = L_cls + 0.5·(L_fact + L_comp)     │")
    print(f"  │          =                      {L_total.item():>6.4f} │")
    print(f"  └─────────────────────────────────────┘")
    print(f"  (All losses are large — random init. They decrease during training.)")

    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 8 — Backward + EMA update")
    # ══════════════════════════════════════════════════════════════════════════

    L_total.backward()

    # show that gradients flowed into f_θ but NOT f_θ̄
    f_theta_grad = sum(
        p.grad.abs().mean().item() for p in f_theta.parameters()
        if p.grad is not None
    )
    f_tbar_grad  = sum(
        p.grad.abs().mean().item() for p in f_tbar.parameters()
        if p.grad is not None
    )
    print(f"  Mean |grad| in f_θ    (context enc) : {f_theta_grad:.6f}  ← gradients flow")
    print(f"  Mean |grad| in f_θ̄   (target  enc) : {f_tbar_grad:.6f}  ← stop-gradient")

    # EMA update: f_θ̄ ← τ·f_θ̄ + (1-τ)·f_θ
    τ = 0.996
    with torch.no_grad():
        for p_bar, p in zip(f_tbar.parameters(), f_theta.parameters()):
            p_bar.data.mul_(τ).add_(p.data, alpha=1 - τ)
    print(f"\n  EMA update: f_θ̄ ← {τ}·f_θ̄ + {1-τ}·f_θ  (target encoder tracks context)")

    print(f"\n{'═'*60}")
    print("  Forward pass complete.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
