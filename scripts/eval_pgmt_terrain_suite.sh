#!/usr/bin/env bash
set -euo pipefail
PGMT_SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_SUITE_LAB=/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source
PGMT_SUITE_ENV=/home/lzp/miniconda3/envs/isaacsim
export OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 LIVESTREAM=0
export PMT_REPO_ROOT="$PGMT_SUITE_ROOT"
export PYTHONPATH="$PGMT_SUITE_LAB/isaaclab:$PGMT_SUITE_LAB/isaaclab_assets:$PGMT_SUITE_LAB/isaaclab_contrib:$PGMT_SUITE_LAB/isaaclab_tasks:$PGMT_SUITE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
cd "$PGMT_SUITE_ROOT"
exec "$PGMT_SUITE_ENV/bin/python" scripts/eval_pgmt_terrain_suite.py --headless \
  --experience "$PGMT_SUITE_ENV/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
  --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
