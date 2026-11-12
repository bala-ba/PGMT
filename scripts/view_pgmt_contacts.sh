#!/usr/bin/env bash
set -euo pipefail
PGMT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_LAB_SOURCE="${PGMT_LAB_SOURCE:-/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source}"
PGMT_SIM_ENV="${PGMT_SIM_ENV:-/home/lzp/miniconda3/envs/isaacsim}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PGMT_LAB_SOURCE}/isaaclab:${PGMT_LAB_SOURCE}/isaaclab_assets:${PGMT_LAB_SOURCE}/isaaclab_contrib:${PGMT_LAB_SOURCE}/isaaclab_tasks:${PGMT_REPO_DIR}"
cd -- "${PGMT_REPO_DIR}"
exec "${PGMT_SIM_ENV}/bin/python" scripts/view_pgmt_contacts.py \
  --device cuda:1 \
  --experience "${PGMT_SIM_ENV}/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
  --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' "$@"
