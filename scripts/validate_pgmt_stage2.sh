#!/usr/bin/env bash
# Full finite P2.4/P2.5 run. Each stage uses a NEW simulator process.
set -euo pipefail
PGMT_AUDIT_ROOT="${1:-logs/pgmt_p245_$(date +%Y%m%d_%H%M%S)}"
PGMT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PGMT_REPO_DIR"
mkdir -p "$PGMT_AUDIT_ROOT"
run_pgmt_audit() {
    local pgmt_name="$1"
    shift
    bash scripts/train_pgmt_stage2.sh --log_dir "$PGMT_AUDIT_ROOT/$pgmt_name" "$@" \
        > "$PGMT_AUDIT_ROOT/${pgmt_name}.log" 2>&1
}
run_pgmt_audit gui_160 --num_envs 160 --max_iterations 25 --acceptance
run_pgmt_audit resume_160 --num_envs 160 --max_iterations 5 --headless --resume "$PGMT_AUDIT_ROOT/gui_160/final.pt"
run_pgmt_audit restore_160 --num_envs 160 --max_iterations 0 --headless --resume "$PGMT_AUDIT_ROOT/resume_160/final.pt"
for pgmt_n in 160 1024; do
    for pgmt_mode in blind scan_only perception; do
        if run_pgmt_audit "cost_${pgmt_n}_${pgmt_mode}" --num_envs "$pgmt_n" --max_iterations 4 --headless --profile "$pgmt_mode"; then
            continue
        fi
        # 160 is mandatory. 1024 is conditional on actual PhysX memory, not
        # Torch allocator estimates. Never report corrupted-scene timings.
        if [[ "$pgmt_n" != 1024 ]]; then exit 1; fi
        /home/lzp/miniconda3/envs/pmt/bin/python - "$PGMT_AUDIT_ROOT" "$pgmt_mode" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
state=json.loads((root/f'cost_1024_{sys.argv[2]}/training_status.json').read_text())
assert state['status']=='failed_physics', state
errors=state.get('physics_errors',[])
assert any('failed to allocate' in e.lower() for e in errors), errors
(root/'scale_limits.json').write_text(json.dumps(dict(num_envs=1024,status='capacity_limited',
    physics_errors=errors,remaining_profiles='not run after common physics OOM'),indent=2)+'\n')
PY
        break
    done
done
# Read the native NumPy 2 RNG pickle in the same runtime that wrote it.
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/report_pgmt_stage2.py "$PGMT_AUDIT_ROOT"
