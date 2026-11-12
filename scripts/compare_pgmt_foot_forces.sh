#!/usr/bin/env bash
# Sequential, paired inference: same checkpoint/motion/seed/patch-local route.
set -euo pipefail
PGMT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PGMT_FORCE_OUTPUT=""
PGMT_FORCE_DEVICE="cuda:0"
PGMT_FORCE_LEVEL=1
PGMT_FORCE_DIRECTION=up
PGMT_FORCE_OFFSET=0
PGMT_FORCE_EPISODES=1
PGMT_FORCE_MAX_STEPS=600
PGMT_FORCE_SEED=1017
PGMT_FORCE_PATCH=5
PGMT_FORCE_EXTRA=()
PGMT_FORCE_CHECKPOINT="$PGMT_REPO_DIR/logs/pgmt_stage2_center2x2/train_03/model_17500.pt"
PGMT_FORCE_MOTION="$PGMT_REPO_DIR/assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz"
while (($#)); do
  case "$1" in
    --record_video_states) PGMT_FORCE_EXTRA+=(--record_video_states); shift ;;
    --log_dir) PGMT_FORCE_OUTPUT="$2"; shift 2 ;;
    --device) PGMT_FORCE_DEVICE="$2"; shift 2 ;;
    --level) PGMT_FORCE_LEVEL="$2"; shift 2 ;;
    --direction) PGMT_FORCE_DIRECTION="$2"; shift 2 ;;
    --spawn_offset_x) PGMT_FORCE_OFFSET="$2"; shift 2 ;;
    --episodes) PGMT_FORCE_EPISODES="$2"; shift 2 ;;
    --max_steps) PGMT_FORCE_MAX_STEPS="$2"; shift 2 ;;
    --seed) PGMT_FORCE_SEED="$2"; shift 2 ;;
    --patch_id) PGMT_FORCE_PATCH="$2"; shift 2 ;;
    --checkpoint) PGMT_FORCE_CHECKPOINT="$2"; shift 2 ;;
    --motion) PGMT_FORCE_MOTION="$2"; shift 2 ;;
    --help|-h)
      echo 'Usage: bash scripts/compare_pgmt_foot_forces.sh --log_dir NEW_DIR [--checkpoint PT] [--motion NPZ] [--device cuda:0] [--level 1] [--direction up] [--spawn_offset_x 0] [--episodes 1] [--max_steps 600] [--seed 1017] [--patch_id 5] [--record_video_states]'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$PGMT_FORCE_OUTPUT" || -e "$PGMT_FORCE_OUTPUT" ]]; then
  echo '--log_dir must name a new output directory' >&2; exit 2
fi
if [[ ! "$PGMT_FORCE_EPISODES" =~ ^[1-9][0-9]*$ || ! "$PGMT_FORCE_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo '--episodes and --max_steps must be positive integers' >&2; exit 2
fi
PGMT_FORCE_OUTPUT="$(realpath -m -- "$PGMT_FORCE_OUTPUT")"
cd -- "$PGMT_REPO_DIR"
mkdir -p -- "$PGMT_FORCE_OUTPUT"
PGMT_FORCE_OUTPUT="$(cd -- "$PGMT_FORCE_OUTPUT" && pwd)"
for PGMT_FORCE_TERRAIN in flat stairs slope; do
  echo "Recording $PGMT_FORCE_TERRAIN -> $PGMT_FORCE_OUTPUT/$PGMT_FORCE_TERRAIN.log"
  bash scripts/play_pgmt_stage2_stairs.sh --headless --record_forces --no_alternate \
    --device "$PGMT_FORCE_DEVICE" --checkpoint "$PGMT_FORCE_CHECKPOINT" --motion "$PGMT_FORCE_MOTION" \
    --terrain "$PGMT_FORCE_TERRAIN" --level "$PGMT_FORCE_LEVEL" --direction "$PGMT_FORCE_DIRECTION" \
    --spawn_offset_x "$PGMT_FORCE_OFFSET" --patch_id "$PGMT_FORCE_PATCH" --seed "$PGMT_FORCE_SEED" \
    --max_episodes "$PGMT_FORCE_EPISODES" --max_steps "$PGMT_FORCE_MAX_STEPS" \
    "${PGMT_FORCE_EXTRA[@]}" \
    --log_dir "$PGMT_FORCE_OUTPUT/$PGMT_FORCE_TERRAIN" > "$PGMT_FORCE_OUTPUT/$PGMT_FORCE_TERRAIN.log" 2>&1
done
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pgmt-force-matplotlib}" \
  /home/lzp/miniconda3/envs/pmt/bin/python scripts/pgmt_force_analysis.py \
  --run "flat=$PGMT_FORCE_OUTPUT/flat" --run "stairs=$PGMT_FORCE_OUTPUT/stairs" \
  --run "slope=$PGMT_FORCE_OUTPUT/slope" --output "$PGMT_FORCE_OUTPUT/comparison"
