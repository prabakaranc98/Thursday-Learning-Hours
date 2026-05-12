#!/usr/bin/env bash
# run_step1.sh — run Step 1 supervised BCE baseline on the A100
# Usage: bash run_step1.sh [DATA_ROOT] [OUT_DIR]
# DATA_ROOT defaults to /data/birdclef-2026
# WANDB_API_KEY must be set in the environment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="${1:-/data/birdclef-2026}"
OUT_DIR="${2:-$REPO_ROOT/outputs/step1}"

# Fall back to system python if venv missing
if [[ ! -f "$VENV_PY" ]]; then
    VENV_PY="$(which python3)"
    echo "[WARN] .venv not found — using system python: $VENV_PY"
    echo "       Run scripts/setup_env.sh first."
fi

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"
LOG="$REPO_ROOT/logs/step1_$(date +%Y%m%d_%H%M%S).log"

echo "==> Step 1 Baseline"
echo "    python    : $VENV_PY"
echo "    DATA_ROOT : $DATA_ROOT"
echo "    OUT_DIR   : $OUT_DIR"
echo "    log       : $LOG"
echo "    WandB     : ${WANDB_API_KEY:+configured}${WANDB_API_KEY:-not set (set WANDB_API_KEY)}"
echo "    GPU       : $($VENV_PY -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo ""

cd "$REPO_ROOT"
BIRDCLEF_DATA_ROOT="$DATA_ROOT" \
BIRDCLEF_OUT_DIR="$OUT_DIR" \
    "$VENV_PY" experiments/step1_baseline/train.py \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        2>&1 | tee "$LOG"

echo ""
echo "==> Done. Checkpoints: $OUT_DIR/checkpoints/"
echo "    Log: $LOG"
