#!/usr/bin/env bash
# Run through scripts/pmt.sh so the conda environment and data paths are set.
set -euo pipefail
PMT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMT_ROOT"
source scripts/pmt_env.sh
smoke_dir="${1:-$PMT_ROOT/logs/smoke/$(date +%Y-%m-%d_%H-%M-%S)}"
if [[ -d "$smoke_dir/flat_train" || -f "$smoke_dir/summary.json" ]]; then
  echo "Refusing to reuse smoke outputs; choose a new directory: $smoke_dir" >&2
  exit 2
fi
mkdir -p "$smoke_dir"
smoke_dir="$(cd "$smoke_dir" && pwd)"
echo "Smoke artifacts: $smoke_dir"
python -m pip check | tee "$smoke_dir/pip-check.log"
python -m pytest tests -q --disable-warnings --tb=short | tee "$smoke_dir/pytest.log"
echo "[smoke] Training flat PPO (5 iterations)..."
python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 8 --max_iterations 5 --seed 42 --headless \
  --log_dir "$smoke_dir/flat_train" > "$smoke_dir/flat_train.log" 2>&1
echo "[smoke] Reloading the new flat checkpoint (200 steps)..."
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 2 --headless --max_steps 200 \
  --resume_path "$smoke_dir/flat_train/model_4.pt" \
  --metrics_file "$smoke_dir/flat_trained_play.json" > "$smoke_dir/flat_trained_play.log" 2>&1
echo "[smoke] Evaluating the pretrained flat checkpoint (300 steps)..."
python scripts/play.py --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 2 --headless --max_steps 300 \
  --resume_path checkpoints/pretrained/multimotionv2_flat.pt \
  --metrics_file "$smoke_dir/flat_pretrained_play.json" > "$smoke_dir/flat_pretrained_play.log" 2>&1
echo "[smoke] Training perceptive token tracker (2 iterations)..."
python scripts/train.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 \
  --motion_file "$PMT_TERRAIN_MOTION_ROOT/terrain_mocaphouse/walk_dance1sub2start/optimized" \
  --num_envs 4 --max_iterations 2 --seed 42 --headless \
  --log_dir "$smoke_dir/perceptive_train" > "$smoke_dir/perceptive_train.log" 2>&1
echo "[smoke] Reloading the new perceptive checkpoint (200 steps)..."
python scripts/play.py --task PMT-PerceptiveMotionTokenTracker-G1-v0 \
  --motion_file "$PMT_TERRAIN_MOTION_ROOT/terrain_mocaphouse/walk_dance1sub2start/optimized" \
  --num_envs 2 --headless --max_steps 200 \
  --resume_path "$smoke_dir/perceptive_train/model_1.pt" \
  --metrics_file "$smoke_dir/perceptive_trained_play.json" > "$smoke_dir/perceptive_trained_play.log" 2>&1
echo "[smoke] Distilling the bundled terrain teacher (2 iterations)..."
python scripts/train.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 \
  "network.teacher_ckpt=$PMT_ROOT/checkpoints/pretrained/walkdance_bigmap_teacher.pt" \
  --num_envs 4 --max_iterations 2 --seed 42 --headless \
  --log_dir "$smoke_dir/distill_train" > "$smoke_dir/distill_train.log" 2>&1
echo "[smoke] Reloading the new distilled student (100 steps)..."
python scripts/play.py --task PMT-Distill-SteppingStone-LatentAnchor-G1-v0 \
  "network.teacher_ckpt=$PMT_ROOT/checkpoints/pretrained/walkdance_bigmap_teacher.pt" \
  --num_envs 2 --headless --max_steps 100 \
  --resume_path "$smoke_dir/distill_train/model_1.pt" \
  --metrics_file "$smoke_dir/distill_trained_play.json" > "$smoke_dir/distill_trained_play.log" 2>&1
echo "[smoke] PPO finetuning the new distilled student (2 iterations)..."
python scripts/train.py --task PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0 \
  "network.base_policy_ckpt=$smoke_dir/distill_train/model_1.pt" \
  --num_envs 4 --max_iterations 2 --seed 42 --headless \
  --log_dir "$smoke_dir/finetune_train" > "$smoke_dir/finetune_train.log" 2>&1
echo "[smoke] Reloading the new finetuned student (100 steps)..."
python scripts/play.py --task PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0 \
  "network.base_policy_ckpt=$smoke_dir/distill_train/model_1.pt" \
  --num_envs 2 --headless --max_steps 100 \
  --resume_path "$smoke_dir/finetune_train/model_1.pt" \
  --metrics_file "$smoke_dir/finetune_trained_play.json" > "$smoke_dir/finetune_trained_play.log" 2>&1
python scripts/summarize_smoke.py "$smoke_dir" --require-smoke-suite
echo "SMOKE PASS: flat PPO, token tracker, teacher distillation, PPO finetuning, checkpoint reloads and finite rollouts."
echo "Artifacts: $smoke_dir"
