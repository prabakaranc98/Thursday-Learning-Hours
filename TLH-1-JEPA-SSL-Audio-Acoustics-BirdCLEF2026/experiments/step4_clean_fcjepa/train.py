"""
Step 4: Clean FC-JEPA on BirdCLEF 2026

Architecture
────────────────────────────────────────────────────────────────
  E_c  context encoder (trainable ViT)  — sees real soundscape windows
  E_t  target encoder (EMA, stop-grad)  — sees label-conditioned clean audio
  P1   SlotAttentionV4 (per-slot mu)    — factorizes context tokens into K slots
  P2   ComposePredictor                 — composes slots into z_mix_hat
  clf  SlotClassifier                   — per-slot classification head
  cls_head  Linear(D, n_cls)           — direct bypass from CLS token

Training (single mode — no BYOL)
────────────────────────────────────────────────────────────────
  For each soundscape window with species_set S:

  Context path (grad):
    spec_ctx → E_c → (cls, tokens)
    slots, acts = P1(tokens, n_slots=n_active)
    z_mix_hat   = P2(slots, acts)
    logits      = cls_head(cls) + clf(slots)

  Target path (no grad, EMA encoder):
    For each s_i ∈ S: z_i = E_t(stitch_clips(s_i))
    If S empty:        z_1 = E_t(white_noise)
    z_comp = normalize(Σ z_i)   ← algebraic, in rep space (not audio space)

  Losses:
    L_cls  = BCEWithLogitsLoss(logits, y)
    L_fact = Hungarian cosine distance(slots[:n_active], {z_i})
    L_comp = 1 − cosine_similarity(z_mix_hat, z_comp)   [skipped if n_active=1]
    L_sig  = SIGReg(cls) + SIGReg(slots)

  Total: L_cls + λ_fact·L_fact + λ_comp·L_comp + λ_sig·L_sig

EMA:  θ_t ← τ·θ_t + (1−τ)·θ_c   per step, τ=0.996
Warm start: --warm_start path/to/step3/best.pt (strict=False)
"""

import argparse
import ast
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader

from dataset import (BirdCLEFStep4Dataset, ClipStep4Dataset,
                     UnlabeledSoundscapeDataset, step4_collate_fn)
from models import FCJEPAv4, sig_reg

# ── WandB ────────────────────────────────────────────────────────────────────

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
        print("[WANDB] WANDB_API_KEY not found — stdout only")
except Exception as _we:
    _USE_WANDB = False
    print(f"[WANDB] unavailable ({_we}) — stdout only")

# ── Config ────────────────────────────────────────────────────────────────────

CFG = dict(
    # Audio
    sample_rate  = 32_000,
    n_mels       = 128,
    n_fft        = 1024,
    hop_length   = 320,
    duration     = 5.0,
    # ViT
    d_model      = 256,
    n_heads      = 4,
    n_layers     = 4,
    patch_size   = 16,
    dropout      = 0.1,
    # Slot attention
    k_max        = 6,         # max slots = max sources per window
    n_slot_iter  = 3,
    # EMA
    ema_tau      = 0.996,
    # Training
    epochs       = 30,
    batch_size   = 32,
    lr           = 5e-4,
    weight_decay = 0.05,
    warmup_epochs = 2,
    # Loss weights
    lambda_fact  = 1.0,
    lambda_comp  = 0.5,
    lambda_sig   = 0.1,
    n_proj_sig   = 64,        # random projections for SIGReg
    # Data
    data_root    = "/kaggle/input/birdclef-2026",
    num_workers  = 4,
    val_fraction = 0.15,
    seed         = 42,
    include_empty_windows = True,
    # Checkpointing
    warm_start   = None,
    out_dir      = "/kaggle/working/checkpoints",
)

# ── Data helpers ──────────────────────────────────────────────────────────────

def find_data_root(cfg_root: str) -> str:
    if os.path.isdir(cfg_root):
        return cfg_root
    base = "/kaggle/input"
    print(f"[DATA] '{cfg_root}' missing — scanning {base}")
    if not os.path.isdir(base):
        return cfg_root
    for sub in ("competitions/birdclef-2026", "birdclef-2026"):
        p = os.path.join(base, sub)
        if os.path.isdir(p):
            return p
    for d in sorted(os.listdir(base)):
        if os.path.isfile(os.path.join(base, d, "sample_submission.csv")):
            return os.path.join(base, d)
    return cfg_root


def load_species_list(data_root: str) -> list[str]:
    for fname in ("taxonomy.csv", "species_list.csv"):
        p = os.path.join(data_root, fname)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            for col in ("primary_label", "species_code", "ebird_code"):
                if col in df.columns:
                    sp = sorted(df[col].dropna().astype(str).unique().tolist())
                    print(f"[DATA] {len(sp)} species from {fname}[{col}]")
                    return sp
    p = os.path.join(data_root, "sample_submission.csv")
    if os.path.isfile(p):
        df  = pd.read_csv(p, nrows=1)
        sp  = sorted(c for c in df.columns if c.lower() != "row_id")
        if sp:
            print(f"[DATA] {len(sp)} species from sample_submission.csv")
            return sp
    audio_dir = os.path.join(data_root, "train_audio")
    if os.path.isdir(audio_dir):
        sp = sorted(d for d in os.listdir(audio_dir)
                    if os.path.isdir(os.path.join(audio_dir, d)))
        print(f"[DATA] {len(sp)} species from train_audio/")
        return sp
    raise RuntimeError(f"Cannot determine species list from {data_root}")

# ── Loss functions ────────────────────────────────────────────────────────────

def hungarian_fact_loss(
    slots: torch.Tensor,
    z_targets: torch.Tensor,
    n_active: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample Hungarian-matched cosine distance between slots and target embeddings.

    slots     : (B, K_max, D)
    z_targets : (K_max, B, D) — unit-normalized target embeddings (from E_t)
    n_active  : (B,)          — number of active sources per sample

    For each sample b, matches slots[b, :M] to z_targets[:M, b] via Hungarian.
    M = n_active[b], capped at min(K_max, n_active[b]).
    """
    losses = []
    B, K_max, D = slots.shape
    for b in range(B):
        M = int(n_active[b].item())
        M = min(M, K_max)
        s = F.normalize(slots[b, :M], dim=-1)   # (M, D)
        t = z_targets[:M, b]                    # (M, D) — already normalized
        cost = 1.0 - (s @ t.T)                  # (M, M) cosine distances
        row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
        losses.append(cost[row, col].mean())
    return torch.stack(losses).mean()


def composition_loss(
    z_mix_hat: torch.Tensor,
    z_targets: torch.Tensor,
    n_active: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine distance between P2 prediction and algebraic composition target.

    Only computed for samples with n_active > 1 (multi-source windows).
    z_comp = normalize(sum of all per-source targets) — computed here.

    z_mix_hat : (B, D)
    z_targets : (K_max, B, D)
    n_active  : (B,)
    """
    B = z_mix_hat.shape[0]
    z_comp_list = []
    for b in range(B):
        M = int(n_active[b].item())
        z_sum = z_targets[:M, b].sum(0)         # (D,)
        z_comp_list.append(F.normalize(z_sum, dim=0))
    z_comp = torch.stack(z_comp_list)           # (B, D)

    sim = F.cosine_similarity(z_mix_hat, z_comp)  # (B,)
    mask = (n_active > 1).float().to(sim.device)
    if mask.sum() == 0:
        return z_mix_hat.new_zeros(())
    return ((1.0 - sim) * mask).sum() / mask.sum()

# ── Validation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def validate(model: FCJEPAv4, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_logits, all_labels = [], []
    for spec_ctx, _, y, __ in loader:
        spec_ctx = spec_ctx.to(device)
        logits   = model(spec_ctx)
        all_logits.append(logits.float().cpu())
        all_labels.append(y.cpu())
    model.train()

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))

    # Macro ROC-AUC over classes that appear in val
    aucs = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:
            try:
                aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0

# ── LR schedule helper ────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default=None)
    parser.add_argument("--warm_start",   default=None, help="Path to Step 3 best.pt")
    parser.add_argument("--out_dir",      default=None)
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--batch_size",   type=int, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--lambda_fact",  type=float, default=None)
    parser.add_argument("--lambda_comp",  type=float, default=None)
    parser.add_argument("--lambda_sig",   type=float, default=None)
    parser.add_argument("--num_workers",  type=int, default=None)
    parser.add_argument("--no_wandb",     action="store_true")
    args = parser.parse_args()

    # Apply CLI overrides
    for k in ("data_root", "warm_start", "out_dir", "epochs", "batch_size",
              "lr", "lambda_fact", "lambda_comp", "lambda_sig", "num_workers"):
        v = getattr(args, k, None)
        if v is not None:
            CFG[k] = v

    use_wandb = _USE_WANDB and not args.no_wandb

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        _cap = torch.cuda.get_device_capability()
        device = torch.device("cuda") if _cap[0] >= 7 else torch.device("cpu")
        print(f"[TRAIN] GPU sm_{_cap[0]}{_cap[1]} → {'CUDA' if _cap[0]>=7 else 'CPU'}")
    else:
        device = torch.device("cpu")
        print("[TRAIN] CPU")

    # Detect A100 and scale batch size
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        CFG["batch_size"] = max(CFG["batch_size"], 64)
        CFG["num_workers"] = max(CFG["num_workers"], 8)
        print(f"[TRAIN] A100 detected → batch={CFG['batch_size']}")

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    use_amp = device.type == "cuda"

    torch.manual_seed(CFG["seed"])
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root    = find_data_root(CFG["data_root"])
    species_list = load_species_list(data_root)
    n_cls        = len(species_list)
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    ds_kw = dict(
        data_root=data_root, species_to_idx=species_to_idx,
        duration=CFG["duration"], sample_rate=CFG["sample_rate"],
        n_mels=CFG["n_mels"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
        val_fraction=CFG["val_fraction"], seed=CFG["seed"],
        include_empty_windows=CFG["include_empty_windows"],
    )
    sc_train = BirdCLEFStep4Dataset(split="train", **ds_kw)
    sc_val   = BirdCLEFStep4Dataset(split="val",   **ds_kw)

    # Curated clips: single-species JEPA examples (~30K samples)
    clip_train = ClipStep4Dataset(split="train", **ds_kw)
    clip_val   = ClipStep4Dataset(split="val",   **ds_kw)

    # All 10,658 unlabeled soundscapes for self-supervised SIGReg training.
    # Subsampled to 8192 windows/epoch to keep epoch time manageable.
    unlabeled_train = UnlabeledSoundscapeDataset(
        data_root=data_root, num_classes=n_cls,
        duration=CFG["duration"], sample_rate=CFG["sample_rate"],
        n_mels=CFG["n_mels"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
        n_per_epoch=8192, seed=CFG["seed"],
    )

    train_ds = ConcatDataset([sc_train, clip_train, unlabeled_train])
    val_ds   = ConcatDataset([sc_val, clip_val])

    if len(train_ds) == 0:
        print("[TRAIN] ERROR: 0 training samples — check data_root and soundscape labels")
        return
    print(f"[TRAIN] Combined: {len(sc_train)} labeled-sc + {len(clip_train)} clips "
          f"+ {len(unlabeled_train)} unlabeled-sc")

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=(device.type == "cuda"),
        collate_fn=step4_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=(device.type == "cuda"),
        collate_fn=step4_collate_fn,
    )
    print(f"[TRAIN] {len(train_ds)} train / {len(val_ds)} val  "
          f"({len(train_loader)} batches/epoch)")

    # ── Model ────────────────────────────────────────────────────────────────
    n_frames = math.ceil(CFG["duration"] * CFG["sample_rate"] / CFG["hop_length"]) + 1
    model = FCJEPAv4(
        n_mels     = CFG["n_mels"],
        n_frames   = n_frames,
        patch_size = CFG["patch_size"],
        D          = CFG["d_model"],
        n_heads    = CFG["n_heads"],
        depth      = CFG["n_layers"],
        dropout    = CFG["dropout"],
        k_max      = CFG["k_max"],
        n_slot_iter = CFG["n_slot_iter"],
        n_cls      = n_cls,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[TRAIN] FCJEPAv4 {n_params:.1f}M params")

    # Warm start from Step 3 (or Step 1) checkpoint
    if CFG.get("warm_start") and os.path.isfile(CFG["warm_start"]):
        try:
            try:
                ckpt = torch.load(CFG["warm_start"], map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(CFG["warm_start"], map_location="cpu")
            state = ckpt.get("model_state", ckpt)
            # Filter out keys with shape mismatches (e.g. slot_mu changed from (1,1,D) to (1,K,D))
            cur = model.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in cur and v.shape == cur[k].shape}
            skipped = [k for k in state if k not in filtered]
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"[TRAIN] warm start: loaded {len(filtered)} keys, "
                  f"skipped {len(skipped)} (shape mismatch), "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
        except Exception as e:
            print(f"[TRAIN] warm start failed: {e} — training from scratch")
    else:
        print("[TRAIN] no warm start — training from scratch")

    # ── Optimizer & schedule ─────────────────────────────────────────────────
    # Separate param groups: encoder gets a lower LR (it's already warm-started)
    enc_params  = list(model.enc_ctx.parameters())
    head_params = (
        list(model.P1.parameters()) +
        list(model.P2.parameters()) +
        list(model.clf.parameters()) +
        list(model.cls_head.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": CFG["lr"] * 0.1},
        {"params": head_params, "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = len(train_loader) * CFG["warmup_epochs"]
    warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                                milestones=[warmup_steps])
    scaler        = GradScaler("cuda", enabled=use_amp)
    bce_loss      = nn.BCEWithLogitsLoss()

    os.makedirs(CFG["out_dir"], exist_ok=True)

    if use_wandb:
        wandb.init(
            project="fc-audio-jepa",
            entity=os.environ.get("WANDB_ENTITY", "karan98"),
            name="step4-clean-fcjepa",
            config=CFG,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc = 0.0
    global_step = 0
    K_MAX = CFG["k_max"]

    for epoch in range(1, CFG["epochs"] + 1):
        unlabeled_train.reshuffle(seed=CFG["seed"] + epoch)   # fresh subsample each epoch
        model.train()
        t0 = time.time()
        ep_cls = ep_fact = ep_comp = ep_sig = ep_total = 0.0
        n_batches = 0

        for spec_ctx, spec_targets_list, y, n_active in train_loader:

            B = spec_ctx.shape[0]
            spec_ctx = spec_ctx.to(device)
            y        = y.to(device)
            n_active = n_active.to(device)

            # Move target specs to device (list of K_MAX tensors, each (B, 1, H, W))
            spec_targets_list = [t.to(device) for t in spec_targets_list]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                # ── Context path (always runs) ────────────────────────────
                cls, tokens = model.encode_ctx(spec_ctx)   # (B, D), (B, N, D)

                # k_batch: number of active slots for this batch
                # 0 means all samples are unlabeled soundscapes — skip JEPA losses
                k_batch = int(n_active.max().item())
                k_run   = max(k_batch, 1)  # always run at least 1 slot for SIGReg
                slots, acts = model.P1(tokens, n_slots=k_run)  # (B, k_run, D)

                # ── Losses that require targets (skip for unlabeled batches) ──
                if k_batch > 0:
                    z_mix_hat = model.P2(slots, acts)             # (B, D)
                    logits    = model.cls_head(cls) + model.clf(slots)

                    # Batch E_t encoding: (k_batch*B, 1, H, W) → (k_batch, B, D)
                    target_batch = torch.cat(spec_targets_list[:k_batch], dim=0)
                    z_all        = model.encode_target(target_batch)
                    z_targets    = z_all.reshape(k_batch, B, -1)

                    L_cls  = bce_loss(logits.float(), y.float())
                    L_fact = hungarian_fact_loss(slots, z_targets, n_active)
                    L_comp = composition_loss(z_mix_hat, z_targets, n_active)
                else:
                    # Unlabeled soundscape batch: no targets, no classification labels
                    L_cls  = slots.new_zeros(())
                    L_fact = slots.new_zeros(())
                    L_comp = slots.new_zeros(())

                # SIGReg runs on all batches — this is the self-supervised signal
                # for unlabeled soundscapes
                cls_norm   = F.normalize(cls.float(), dim=-1)
                slots_norm = F.normalize(slots.float().reshape(B * k_run, -1), dim=-1)
                L_sig = (
                    sig_reg(cls_norm,   n_proj=CFG["n_proj_sig"]) +
                    sig_reg(slots_norm, n_proj=CFG["n_proj_sig"])
                )

                L = (L_cls
                     + CFG["lambda_fact"] * L_fact
                     + CFG["lambda_comp"] * L_comp
                     + CFG["lambda_sig"]  * L_sig)

            scaler.scale(L).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            model.ema_update(tau=CFG["ema_tau"])

            l_cls_v  = L_cls.item()
            l_fact_v = L_fact.item()
            l_comp_v = L_comp.item() if isinstance(L_comp, torch.Tensor) else float(L_comp)
            l_sig_v  = L_sig.item()
            l_tot_v  = L.item()
            lr_step  = optimizer.param_groups[1]["lr"]

            ep_cls   += l_cls_v
            ep_fact  += l_fact_v
            ep_comp  += l_comp_v
            ep_sig   += l_sig_v
            ep_total += l_tot_v
            n_batches += 1
            global_step += 1

            if use_wandb:
                wandb.log({
                    "step/loss":  l_tot_v, "step/cls":  l_cls_v,
                    "step/fact":  l_fact_v, "step/comp": l_comp_v,
                    "step/sig":   l_sig_v,  "step/lr":   lr_step,
                }, step=global_step)

        ep_cls   /= n_batches
        ep_fact  /= n_batches
        ep_comp  /= n_batches
        ep_sig   /= n_batches
        ep_total /= n_batches
        elapsed   = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────────
        val_auc = validate(model, val_loader, device) if len(val_ds) > 0 else 0.0

        lr_now = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"loss={ep_total:.4f}  cls={ep_cls:.4f}  fact={ep_fact:.4f}  "
            f"comp={ep_comp:.4f}  sig={ep_sig:.4f}  "
            f"val_auc={val_auc:.4f}  lr={lr_now:.2e}  t={elapsed:.0f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss/total": ep_total, "loss/cls": ep_cls,
                "loss/fact": ep_fact,   "loss/comp": ep_comp,
                "loss/sig": ep_sig,     "val/auc": val_auc,
                "lr": lr_now,
            })

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auc": val_auc,
            "cfg": CFG,
            "species": species_list,
        }
        torch.save(ckpt, os.path.join(CFG["out_dir"], "last.pt"))
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt, os.path.join(CFG["out_dir"], "best.pt"))
            print(f"  ✓ new best AUC={best_auc:.4f} — saved best.pt")

    print(f"\nDone. Best val_auc={best_auc:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
