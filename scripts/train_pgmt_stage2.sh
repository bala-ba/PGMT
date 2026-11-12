#!/usr/bin/env bash
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_LAB_SOURCE=/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source
PGMT_GUI_ENV=/home/lzp/miniconda3/envs/isaacsim
PGMT_GUI_EXPERIENCE="$PGMT_GUI_ENV/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
PGMT_ROBOT_USD="$PGMT_REPO_DIR/logs/pgmt_p1_acceptance/native_robot/main.usd"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PGMT_LAB_SOURCE/isaaclab:$PGMT_LAB_SOURCE/isaaclab_assets:$PGMT_LAB_SOURCE/isaaclab_contrib:$PGMT_LAB_SOURCE/isaaclab_tasks:$PGMT_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PMT_REPO_ROOT="$PGMT_REPO_DIR"
exec "$PGMT_GUI_ENV/bin/python" "$PGMT_REPO_DIR/scripts/train_pgmt_stage2.py" \
  --isaacsim6_compat --robot_usd "$PGMT_ROBOT_USD" --experience "$PGMT_GUI_EXPERIENCE" \
  --device cuda:1 --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
