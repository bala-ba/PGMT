#!/usr/bin/env bash
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PGMT_REPO_DIR"
PGMT_LAB_SOURCE="${PGMT_LAB_SOURCE:-/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source}"
PGMT_GUI_ENV="${PGMT_GUI_ENV:-/home/lzp/miniconda3/envs/isaacsim}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PGMT_LAB_SOURCE/isaaclab:$PGMT_LAB_SOURCE/isaaclab_assets:$PGMT_LAB_SOURCE/isaaclab_contrib:$PGMT_LAB_SOURCE/isaaclab_tasks:$PGMT_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PMT_REPO_ROOT="$PGMT_REPO_DIR"
exec "$PGMT_GUI_ENV/bin/python" "$PGMT_REPO_DIR/scripts/diagnose_pgmt_physx_memory.py" \
  --experience "$PGMT_GUI_ENV/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
  --device cuda:1 --headless --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
