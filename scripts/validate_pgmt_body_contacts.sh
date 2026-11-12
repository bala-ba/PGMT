#!/usr/bin/env bash
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PGMT_REPO_DIR"
PGMT_LAB_SOURCE=/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source
PGMT_SIM_ENV=/home/lzp/miniconda3/envs/isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PGMT_LAB_SOURCE/isaaclab:$PGMT_LAB_SOURCE/isaaclab_assets:$PGMT_LAB_SOURCE/isaaclab_contrib:$PGMT_LAB_SOURCE/isaaclab_tasks:$PGMT_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PMT_REPO_ROOT="$PGMT_REPO_DIR"
exec "$PGMT_SIM_ENV/bin/python" scripts/validate_pgmt_body_contacts.py \
  --experience "$PGMT_SIM_ENV/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
  --headless --device cuda:0 --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
