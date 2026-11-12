#!/usr/bin/env bash
set -euo pipefail
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PGMT_REPO_DIR/scripts/train_pgmt_stage2.sh" --task pgmt_stage2_terrain "$@"
