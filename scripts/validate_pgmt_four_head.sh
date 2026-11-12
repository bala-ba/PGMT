#!/usr/bin/env bash
# Each migration/resume is a new simulator process, and all runs are finite.
set -euo pipefail
PGMT_AUDIT_ROOT="${1:-logs/pgmt_p3_four_head_$(date +%Y%m%d_%H%M%S)}"
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PGMT_REPO_DIR"
mkdir -p "$PGMT_AUDIT_ROOT"
run_pgmt_four_head() {
    local pgmt_run_name="$1"
    shift
    bash scripts/train_pgmt_terrain.sh --headless --device cuda:0 --num_envs 160 --save_interval 1 \
        --log_dir "$PGMT_AUDIT_ROOT/$pgmt_run_name" "$@" > "$PGMT_AUDIT_ROOT/$pgmt_run_name.log" 2>&1
}
run_pgmt_four_head stage1_160_01 --max_iterations 3
run_pgmt_four_head resume_160_01 --max_iterations 2 --resume "$PGMT_AUDIT_ROOT/stage1_160_01/final.pt"
run_pgmt_four_head restore_160_01 --max_iterations 0 --resume "$PGMT_AUDIT_ROOT/resume_160_01/final.pt"
run_pgmt_four_head p2_160_01 --max_iterations 0 --initialize-from logs/pgmt_p245/resume_160/final.pt
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/report_pgmt_four_head.py "$PGMT_AUDIT_ROOT"
