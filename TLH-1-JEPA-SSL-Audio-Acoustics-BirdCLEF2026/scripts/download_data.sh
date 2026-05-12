#!/usr/bin/env bash
# download_data.sh — download BirdCLEF 2026 via kagglehub (supports KGAT_ tokens)
# Usage: bash download_data.sh [DATA_ROOT]
# DATA_ROOT defaults to /data/birdclef-2026
# Requires: KAGGLE_API_TOKEN env var set (export KAGGLE_API_TOKEN=KGAT_...)
set -euo pipefail

DATA_ROOT="${1:-/data/birdclef-2026}"
VENV_PY="${VENV_PY:-/workspace/tlh1/.venv/bin/python}"
COMP="birdclef-2026"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[ERROR] KAGGLE_API_TOKEN not set. Export it before running:"
    echo "  export KAGGLE_API_TOKEN=KGAT_..."
    exit 1
fi

echo "==> Downloading $COMP to $DATA_ROOT"
mkdir -p "$DATA_ROOT"

"$VENV_PY" - <<PYEOF
import kagglehub, os, sys
token = os.environ['KAGGLE_API_TOKEN']
print(f"[kagglehub] auth token: {token[:12]}...")
print(f"[kagglehub] downloading birdclef-2026 → $DATA_ROOT")
path = kagglehub.competition_download(
    "birdclef-2026",
    output_dir="$DATA_ROOT",
    force_download=False,
)
print(f"[kagglehub] done: {path}")
PYEOF

echo ""
echo "==> Data layout:"
du -sh "$DATA_ROOT"/*/  2>/dev/null | sort -rh | head -12
echo ""
echo "Key files:"
wc -l "$DATA_ROOT/train_soundscapes_labels.csv" 2>/dev/null || true
wc -l "$DATA_ROOT/train.csv" 2>/dev/null || true
ls "$DATA_ROOT/train_audio/" 2>/dev/null | wc -l | xargs echo "  train_audio/ species dirs:"
ls "$DATA_ROOT/train_soundscapes/" 2>/dev/null | wc -l | xargs echo "  train_soundscapes/ files:"
echo ""
echo "Download complete. DATA_ROOT=$DATA_ROOT"
