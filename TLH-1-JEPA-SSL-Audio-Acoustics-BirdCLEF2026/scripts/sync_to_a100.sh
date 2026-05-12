#!/usr/bin/env bash
# sync_to_a100.sh — rsync repo to A100, excluding data and build artifacts
# Usage: bash sync_to_a100.sh [REMOTE]
# REMOTE defaults to root@217.18.55.107
set -euo pipefail

REMOTE="${1:-root@217.18.55.107}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="/workspace/tlh1"

echo "==> Syncing $REPO_ROOT → $REMOTE:$REMOTE_DIR"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='artifacts/pdf/' \
    --exclude='artifacts/png/' \
    --exclude='logs/*.log' \
    --exclude='outputs/' \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    "$REPO_ROOT/" \
    "$REMOTE:$REMOTE_DIR/"

echo ""
echo "==> Sync complete. On remote:"
echo "    cd $REMOTE_DIR"
echo "    bash scripts/setup_env.sh"
echo "    bash scripts/download_data.sh /data/birdclef-2026"
echo "    WANDB_API_KEY=<key> bash scripts/run_step1.sh /data/birdclef-2026"
