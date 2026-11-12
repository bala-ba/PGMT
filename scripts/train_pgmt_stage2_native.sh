#!/usr/bin/env bash
# Native pmt / Isaac Sim 5.1 headless entry point.
# For this workstation's GUI use train_pgmt_terrain.sh (Sim 6, display GPU 1).
# Sim 5.1 GUI still crashes in native renderer startup; see docs/PGMT_GUI_STARTUP.zh-CN.md.
# Single GPU: --device cuda:1. Multiple GPUs: --gpus 0,1 (num_envs is the total).
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_NATIVE_ENV=/home/lzp/miniconda3/envs/pmt
PGMT_LAB_SOURCE="$PGMT_NATIVE_ENV/lib/python3.11/site-packages/isaaclab/source"
export OMNI_KIT_ACCEPT_EULA=YES
export PMT_REPO_ROOT="$PGMT_REPO_DIR"
export PYTHONPATH="$PGMT_LAB_SOURCE/isaaclab:$PGMT_LAB_SOURCE/isaaclab_assets:$PGMT_LAB_SOURCE/isaaclab_contrib:$PGMT_LAB_SOURCE/isaaclab_tasks:$PGMT_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# The pmt environment's NCCL 2.26.2 has a Blackwell multi-GPU kernel bug.
# Use the existing 2.27.5 runtime only in distributed child processes; no Conda
# package changes, and no preload on the single-GPU path. Override as needed.
export PGMT_NCCL_LIBRARY="${PGMT_NCCL_LIBRARY-/home/lzp/miniconda3/envs/isaacsim/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"
exec "$PGMT_NATIVE_ENV/bin/python" "$PGMT_REPO_DIR/scripts/train_pgmt_stage2.py" \
  --robot_usd "$PGMT_REPO_DIR/logs/pgmt_p1_acceptance/native_robot/main.usd" \
  --task pgmt_stage2_terrain --headless --device cuda:0 --performance all "$@"
