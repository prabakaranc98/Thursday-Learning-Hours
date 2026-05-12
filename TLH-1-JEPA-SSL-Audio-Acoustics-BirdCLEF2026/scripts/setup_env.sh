#!/usr/bin/env bash
# setup_env.sh — create a uv virtual environment for TLH-1 with CUDA-compatible PyTorch
# Usage: bash setup_env.sh
# Idempotent: safe to re-run; skips steps already done.
set -euo pipefail

VENV_DIR="/workspace/tlh1/.venv"
UV="$(which uv 2>/dev/null || echo /root/miniconda3/envs/py3.10/bin/uv)"

echo "==> uv: $($UV --version)"
echo "==> [1/4] Creating venv at $VENV_DIR (Python 3.10)"
$UV venv "$VENV_DIR" --python 3.10 --seed 2>/dev/null || echo "  venv already exists"

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "==> [2/4] Installing PyTorch 2.6.0+cu126"
$UV pip install \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --python "$PY"

echo "==> [3/4] Installing project dependencies"
$UV pip install \
    "numpy>=1.24,<2.0" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "scipy>=1.11" \
    "soundfile>=0.12" \
    "librosa>=0.10" \
    "wandb>=0.17" \
    "kagglehub>=0.4.1" \
    "pyyaml>=6.0" \
    "tqdm>=4.65" \
    --python "$PY"

echo "==> [4/4] Verifying torch + CUDA"
"$PY" - <<'EOF'
import torch, torchaudio
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
print(f"  torch     : {torch.__version__}")
print(f"  torchaudio: {torchaudio.__version__}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  sm_cap    : sm_{cap[0]}{cap[1]}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  dtype     : {'bfloat16' if cap[0] >= 8 else 'float16'}")
EOF

echo ""
echo "Setup complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Or run directly: $PY experiments/step1_baseline/train.py ..."
