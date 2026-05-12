#!/usr/bin/env bash
# start_step3.sh — launch Step 3 FC Audio JEPA training on the A100
# Usage: bash start_step3.sh [DATA_ROOT] [OUT_DIR] [STEP1_CKPT]
# Defaults to the standard paths from the A100 setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="${1:-/data/birdclef-2026}"
OUT_DIR="${2:-$REPO_ROOT/outputs/step3}"
STEP1_CKPT="${3:-$REPO_ROOT/outputs/step1/checkpoints/best.pt}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] WANDB_API_KEY not set — training will run without WandB"
fi

echo "==> Step 3: FC Audio JEPA"
echo "    data    : $DATA_ROOT"
echo "    out     : $OUT_DIR"
echo "    step1   : $STEP1_CKPT"
echo ""

# WandB login
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$VENV_PY" -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" \
        && echo "[WANDB] logged in"
fi

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"
LOG="$REPO_ROOT/logs/step3_$(date +%Y%m%d_%H%M%S).log"

echo "    GPU    : $($VENV_PY -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo "    log    : $LOG"
echo ""

cd "$REPO_ROOT"

STEP1_ARG=""
if [[ -f "$STEP1_CKPT" ]]; then
    echo "[INFO] Loading Step 1 weights from $STEP1_CKPT"
    STEP1_ARG="--step1_ckpt $STEP1_CKPT"
else
    echo "[WARN] Step 1 checkpoint not found at $STEP1_CKPT — training from scratch"
fi

"$VENV_PY" experiments/step3_fc_jepa/train.py \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR" \
    $STEP1_ARG \
    2>&1 | tee "$LOG"

echo ""
echo "==> Step 3 complete."
echo "    Checkpoints : $OUT_DIR/checkpoints/"
echo "    Log         : $LOG"
