#!/usr/bin/env bash
# submit_step3.sh — run inference from a checkpoint and submit to Kaggle
#
# Usage:  bash submit_step3.sh [CKPT_PATH] [MESSAGE]
#   CKPT_PATH  defaults to /workspace/tlh1/outputs/step3/checkpoints/best.pt
#              (also accepts step1 checkpoints)
#   MESSAGE    submission description (default: "fc-jepa-step3")
#
# Requires: KAGGLE_API_TOKEN env var (KGAT_ format)
# Submits to competition: birdclef-2026
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="${BIRDCLEF_DATA_ROOT:-/data/birdclef-2026}"
CKPT="${1:-$REPO_ROOT/outputs/step3/checkpoints/best.pt}"
MSG="${2:-fc-jepa-step3-$(date +%Y%m%d_%H%M)}"
OUT_DIR="$REPO_ROOT/outputs/submissions/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "==> FC JEPA Submission"
echo "    ckpt : $CKPT"
echo "    data : $DATA_ROOT"
echo "    out  : $OUT_DIR"
echo "    msg  : $MSG"
echo ""

if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] checkpoint not found: $CKPT"; exit 1
fi
if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[ERROR] KAGGLE_API_TOKEN not set"; exit 1
fi

# Run inference: load ckpt, slide over test_soundscapes, write submission.csv
"$VENV_PY" - <<PYEOF
import os, glob, math, torch, torchaudio, pandas as pd, numpy as np
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.nn as nn, copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "$DATA_ROOT"
CKPT_PATH = "$CKPT"
OUT_DIR   = "$OUT_DIR"

# ── Minimal model classes (must match training code) ──────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class ViTBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(h, d), nn.Dropout(dropout))
    def forward(self, x):
        y, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + y
        return x + self.mlp(self.n2(x))

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=128, n_frames=501, patch_size=16, d=256,
                 n_heads=4, depth=4, dropout=0.1):
        super().__init__()
        n_h = n_mels // patch_size
        n_w = math.ceil(n_frames / patch_size)
        N = n_h * n_w
        self.patch_embed = PatchEmbed(d, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed   = nn.Parameter(torch.zeros(1, N + 1, d))
        self.blocks      = nn.ModuleList([ViTBlock(d, n_heads, dropout=dropout) for _ in range(depth)])
        self.norm        = nn.LayerNorm(d)
    def forward(self, x, return_tokens=False):
        t = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        t = torch.cat([cls, t], dim=1)
        t = t + self.pos_embed[:, :t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        if return_tokens:
            return t[:, 0], t[:, 1:]
        return t[:, 0]

class SlotAttention(nn.Module):
    def __init__(self, K=8, D=256, n_iter=3, eps=1e-8):
        super().__init__()
        self.K, self.D, self.n_iter, self.eps = K, D, n_iter, eps
        self.scale = D ** -0.5
        self.slot_mu        = nn.Parameter(torch.zeros(1, 1, D))
        self.slot_log_sigma = nn.Parameter(torch.full((1, 1, D), -2.0))
        self.norm_in    = nn.LayerNorm(D)
        self.norm_slots = nn.LayerNorm(D)
        self.norm_mlp   = nn.LayerNorm(D)
        self.to_k = nn.Linear(D, D, bias=False)
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.gru  = nn.GRUCell(D, D)
        self.mlp  = nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
    def forward(self, inputs):
        B = inputs.shape[0]
        normed = self.norm_in(inputs)
        k, v = self.to_k(normed), self.to_v(normed)
        sigma = self.slot_log_sigma.exp().expand(B, self.K, self.D)
        slots = self.slot_mu.expand(B, self.K, self.D) + sigma * torch.randn_like(sigma)
        for _ in range(self.n_iter):
            prev = slots
            q    = self.to_q(self.norm_slots(slots))
            dots = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = dots.softmax(dim=1)
            attn_n = attn / (attn.sum(-1, keepdim=True) + self.eps)
            upd  = torch.einsum("bkn,bnd->bkd", attn_n, v)
            slots = self.gru(upd.reshape(B*self.K, self.D),
                             prev.reshape(B*self.K, self.D)).reshape(B, self.K, self.D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn.sum(-1)

class ComposePredictor(nn.Module):
    def __init__(self, D=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D))
    def forward(self, slots, acts):
        w = acts.softmax(-1).unsqueeze(-1)
        return self.mlp((slots * w).sum(1))

class SlotClassifier(nn.Module):
    def __init__(self, D=256, n_cls=234):
        super().__init__()
        self.head = nn.Linear(D, n_cls)
    def forward(self, slots):
        per_slot = self.head(slots)
        return (per_slot.max(dim=1).values + per_slot.mean(dim=1)) * 0.5

class FCJEPAModel(nn.Module):
    def __init__(self, n_mels, n_frames, patch_size, D, n_heads, depth, dropout,
                 n_slots, n_slot_iter, n_cls):
        super().__init__()
        enc_args = (n_mels, n_frames, patch_size, D, n_heads, depth, dropout)
        self.enc_ctx = AudioEncoder(*enc_args)
        self.enc_tgt = copy.deepcopy(self.enc_ctx)
        for p in self.enc_tgt.parameters():
            p.requires_grad_(False)
        self.P1  = SlotAttention(n_slots, D, n_slot_iter)
        self.P2  = ComposePredictor(D)
        self.clf = SlotClassifier(D, n_cls)
        self.cls_head = nn.Linear(D, n_cls)
    def forward(self, x):
        cls, tokens = self.enc_ctx(x, return_tokens=True)
        slots, acts = self.P1(tokens)
        slot_logits = self.clf(slots)
        cls_logits  = self.cls_head(cls)
        return cls_logits + slot_logits, slot_logits, cls_logits, slots, acts, cls

# ── Check if this is a step1 or step3 checkpoint ──────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
state  = ckpt.get("model_state", ckpt)
cfg    = ckpt.get("cfg", {})
species_list = ckpt.get("species", None)
is_step1 = any(k.startswith("head.") for k in state) and not any(k.startswith("clf.") for k in state)
print(f"[INFER] checkpoint: {'step1 (BaselineClassifier)' if is_step1 else 'step3 (FCJEPAModel)'}")
print(f"[INFER] species: {len(species_list) if species_list else 'unknown'}")

sr         = cfg.get("sample_rate", 32000)
n_mels     = cfg.get("n_mels", 128)
n_fft      = cfg.get("n_fft", 1024)
hop_length = cfg.get("hop_length", 320)
duration   = cfg.get("duration", 5.0)
n_frames   = math.ceil(duration * sr / hop_length) + 1
patch_size = cfg.get("patch_size", 16)
D          = cfg.get("d_model", 256)
n_heads    = cfg.get("n_heads", 4)
depth      = cfg.get("n_layers", 4)
dropout    = cfg.get("dropout", 0.1)
n_slots    = cfg.get("n_slots", 8)
n_slot_iter = cfg.get("n_slot_iter", 3)
n_cls      = len(species_list) if species_list else 234

n_samples = int(sr * duration)
win_sec   = int(duration)

if is_step1:
    # Load as BaselineClassifier equivalent using FCJEPAModel with cls_head only
    model = FCJEPAModel(n_mels, n_frames, patch_size, D, n_heads, depth, dropout,
                        n_slots, n_slot_iter, n_cls).to(device)
    enc_state  = {k[4:]: v for k, v in state.items() if k.startswith("enc.")}
    head_state = {k[5:]: v for k, v in state.items() if k.startswith("head.")}
    model.enc_ctx.load_state_dict(enc_state, strict=False)
    model.cls_head.load_state_dict(head_state, strict=False)
else:
    model = FCJEPAModel(n_mels, n_frames, patch_size, D, n_heads, depth, dropout,
                        n_slots, n_slot_iter, n_cls).to(device)
    model.load_state_dict(state, strict=False)

model.eval()

mel_tf = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
db_tf  = T.AmplitudeToDB(top_db=80)

sub_tmpl = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
species_cols = [c for c in sub_tmpl.columns if c != "row_id"]
col_idx = {col: (species_list.index(col) if species_list and col in species_list else -1)
           for col in species_cols}
print(f"[INFER] submission template: {len(sub_tmpl)} rows, {len(species_cols)} species cols")

preds = {}
test_dir = os.path.join(DATA_ROOT, "test_soundscapes")
files = sorted(glob.glob(os.path.join(test_dir, "*.ogg")))
print(f"[INFER] {len(files)} test soundscape files")

amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

with torch.no_grad():
    for fpath in files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        wav, file_sr = torchaudio.load(fpath)
        if file_sr != sr:
            wav = T.Resample(file_sr, sr)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        start, end_sec = 0, win_sec
        while start < wav.shape[1]:
            chunk = wav[:, start: start + n_samples]
            if chunk.shape[1] < n_samples:
                chunk = F.pad(chunk, (0, n_samples - chunk.shape[1]))
            spec = db_tf(mel_tf(chunk))
            spec = (spec + 80.0) / 80.0
            spec = spec.clamp(0.0, 1.0).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
                logits, _, _, _, _, _ = model(spec)
            probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy()
            preds[f"{stem}_{end_sec}"] = probs
            start   += n_samples
            end_sec += win_sec

rows = []
for rid in sub_tmpl["row_id"]:
    p   = preds.get(rid, np.zeros(n_cls, dtype=np.float32))
    row = {"row_id": rid}
    for col in species_cols:
        idx = col_idx[col]
        row[col] = float(p[idx]) if idx >= 0 and idx < len(p) else 0.0
    rows.append(row)

sub_df   = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "submission.csv")
sub_df.to_csv(out_path, index=False)
print(f"[INFER] saved → {out_path}  ({len(sub_df)} rows × {len(species_cols)} cols)")
PYEOF

SUB_CSV="$OUT_DIR/submission.csv"
if [[ ! -f "$SUB_CSV" ]]; then
    echo "[ERROR] inference failed — submission.csv not generated"; exit 1
fi
echo ""
echo "==> Submitting to Kaggle (birdclef-2026)..."
echo "    file : $SUB_CSV"
echo "    msg  : $MSG"

# Submit via Kaggle REST API with KGAT_ token
"$VENV_PY" - <<PYEOF
import os, sys, requests

token  = os.environ.get("KAGGLE_API_TOKEN", "")
if not token:
    print("[ERROR] KAGGLE_API_TOKEN not set"); sys.exit(1)

csv_path = "$SUB_CSV"
comp     = "birdclef-2026"
desc     = "$MSG"

url = f"https://www.kaggle.com/api/v1/competitions/{comp}/submissions"
headers = {"Authorization": f"Bearer {token}"}

with open(csv_path, "rb") as f:
    resp = requests.post(url, headers=headers,
                         files={"file": ("submission.csv", f, "text/csv")},
                         data={"submissionDescription": desc})

if resp.status_code in (200, 201):
    data = resp.json()
    print(f"[SUBMIT] ✓ accepted  id={data.get('id', '?')}  "
          f"status={data.get('status', '?')}")
    print(f"[SUBMIT] score={data.get('publicScore', 'pending...')}")
    print(f"[SUBMIT] view: https://www.kaggle.com/competitions/{comp}/submissions")
else:
    print(f"[SUBMIT] ✗ HTTP {resp.status_code}: {resp.text[:500]}")
    sys.exit(1)
PYEOF
