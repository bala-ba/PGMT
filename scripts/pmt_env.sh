#!/usr/bin/env bash
# Source from any working directory after `conda activate pmt`.
PMT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PMT_REPO_ROOT
export OMNI_KIT_ACCEPT_EULA=YES
export PMT_ASSET_DIR="${PMT_ASSET_DIR:-$PMT_REPO_ROOT/pmt_tasks/assets}"
export PMT_MOTION_ROOT="${PMT_MOTION_ROOT:-$PMT_REPO_ROOT/assets/motions}"
export PMT_TERRAIN_MOTION_ROOT="${PMT_TERRAIN_MOTION_ROOT:-$PMT_REPO_ROOT/assets/motions}"
export PMT_TERRAIN_ASSET_DIR="${PMT_TERRAIN_ASSET_DIR:-$PMT_REPO_ROOT/assets/terrain}"
export PMT_MULTIMOTION_FLAT_MOTION="${PMT_MULTIMOTION_FLAT_MOTION:-$PMT_MOTION_ROOT/terrain_mocaphouse/walk_dance1sub2start/raw}"
export PYTHONPATH="$PMT_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
