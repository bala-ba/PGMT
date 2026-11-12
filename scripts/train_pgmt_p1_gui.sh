#!/usr/bin/env bash
set -euo pipefail

# Use the working Sim 6 GUI with the same Python task/manager implementation as
# the accepted PMT runtime. Do not inherit the legacy Kit file's pinned plugins.
PGMT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_LAB_SOURCE="${PGMT_LAB_SOURCE:-/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source}"
PGMT_GUI_ENV="${PGMT_GUI_ENV:-/home/lzp/miniconda3/envs/isaacsim}"
PGMT_GUI_EXPERIENCE="${PGMT_GUI_EXPERIENCE:-${PGMT_GUI_ENV}/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit}"
PGMT_ROBOT_USD="${PGMT_ROBOT_USD:-${PGMT_REPO_DIR}/logs/pgmt_p1_acceptance/native_robot/main.usd}"
if [[ ! -f "${PGMT_ROBOT_USD}" ]]; then
    echo "Missing audited native G1 USD: ${PGMT_ROBOT_USD}" >&2
    exit 1
fi
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="${PGMT_LAB_SOURCE}/isaaclab:${PGMT_LAB_SOURCE}/isaaclab_assets:${PGMT_LAB_SOURCE}/isaaclab_contrib:${PGMT_LAB_SOURCE}/isaaclab_tasks:${PGMT_REPO_DIR}"
cd -- "${PGMT_REPO_DIR}"
exec "${PGMT_GUI_ENV}/bin/python" scripts/train_pgmt_p1.py \
    --isaacsim6_compat --robot_usd "${PGMT_ROBOT_USD}" \
    --experience "${PGMT_GUI_EXPERIENCE}" --device cuda:1 \
    --kit_args='--/exts/omni.syntheticdata/menubar/visible=false' \
    "$@"
