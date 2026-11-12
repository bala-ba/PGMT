#!/usr/bin/env bash
# Actual Isaac Sim 6 GUI on this workstation's display-connected GPU.
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_LAB_SOURCE=/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source
PGMT_GUI_ENV=/home/lzp/miniconda3/envs/isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=0 LIVESTREAM=0
export PMT_REPO_ROOT="$PGMT_REPO_DIR"
export PYTHONPATH="$PGMT_LAB_SOURCE/isaaclab:$PGMT_LAB_SOURCE/isaaclab_assets:$PGMT_LAB_SOURCE/isaaclab_contrib:$PGMT_LAB_SOURCE/isaaclab_tasks:$PGMT_REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$PGMT_REPO_DIR"
exec "$PGMT_GUI_ENV/bin/python" scripts/play_pgmt_stage2_stairs.py \
  --device cuda:1 \
  --experience "$PGMT_GUI_ENV/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
  --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
