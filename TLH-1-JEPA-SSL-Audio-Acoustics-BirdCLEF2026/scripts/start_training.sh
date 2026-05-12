#!/usr/bin/env bash
# start_training.sh — wait for data download to finish, then launch Step 1 training
# Usage: WANDB_API_KEY=<key> bash start_training.sh [DATA_ROOT] [OUT_DIR]
# Run this AFTER download_data.sh has been started (it polls for completion).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="${1:-/data/birdclef-2026}"
OUT_DIR="${2:-$REPO_ROOT/outputs/step1}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] WANDB_API_KEY not set — training will run without WandB monitoring"
fi

echo "==> Waiting for data at $DATA_ROOT/train.csv ..."
until [[ -f "$DATA_ROOT/train.csv" ]]; do
    # Show download progress if log exists
    DLLOG=$(ls "$REPO_ROOT/logs/download_"*.log 2>/dev/null | tail -1)
    if [[ -n "$DLLOG" ]]; then
        echo -n "  $(tail -1 "$DLLOG" | tr -d '\r' | sed 's/.*\r//')"
        echo ""
    fi
    sleep 30
done
echo ""
echo "==> Data ready. Starting training..."
echo ""

# WandB login
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$VENV_PY" -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" 2>&1 \
        && echo "[WANDB] logged in"
fi

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"
LOG="$REPO_ROOT/logs/step1_$(date +%Y%m%d_%H%M%S).log"

echo "==> Step 1 Baseline"
echo "    GPU    : $($VENV_PY -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo "    VRAM   : $($VENV_PY -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")' 2>/dev/null || echo 'N/A')"
echo "    OUT    : $OUT_DIR"
echo "    log    : $LOG"
echo ""

cd "$REPO_ROOT"
"$VENV_PY" experiments/step1_baseline/train.py \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$LOG"

echo ""
echo "==> Training complete."
echo "    Checkpoints : $OUT_DIR/checkpoints/"
echo "    Log         : $LOG"
