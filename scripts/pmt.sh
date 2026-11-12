#!/usr/bin/env bash
# PMT's isolated environment and the existing tutorial VNC display.
set -euo pipefail
PMT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PMT_CONDA_ROOT="${PMT_CONDA_ROOT:-$HOME/miniconda3}"
PMT_ENV="${PMT_ENV:-pmt}"
source "$PMT_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$PMT_ENV"
source "$PMT_ROOT/scripts/pmt_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$PMT_ROOT"
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  train|play)
    exec python "scripts/$mode.py" "$@"
    ;;
  gui-train|gui-play)
    tutorial="${PMT_TUTORIAL_DIR:-$HOME/projects/tutorial}"
    bash "$tutorial/isaacsim_gui.sh" display --no-hints
    # Isaac Sim physics stays headless; the separate Mesa viewer uses tutorial's Xvfb.
    unset DISPLAY
    exec python "scripts/${mode#gui-}.py" --headless --state_viz \
      --state_viz_display ":${DISPLAY_NUM:-99}" --follow_robot "$@"
    ;;
  test)
    exec python -m pytest tests "$@"
    ;;
  mujoco-eval)
    if [[ " $* " == *"--perception depth"* || " $* " == *"--perception=depth"* ]]; then
      export MUJOCO_GL="${MUJOCO_GL:-egl}"
    fi
    exec python scripts/mujoco_eval_student.py "$@"
    ;;
  mujoco-live)
    tutorial="${PMT_TUTORIAL_DIR:-$HOME/projects/tutorial}"
    bash "$tutorial/isaacsim_gui.sh" display --no-hints
    export DISPLAY=":${DISPLAY_NUM:-99}"
    export MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa
    unset LD_LIBRARY_PATH LD_PRELOAD
    exec python scripts/mujoco_live_student.py "$@"
    ;;
  mujoco-stop)
    exec python scripts/stop_mujoco_live.py "$@"
    ;;
  depth-replay)
    exec python scripts/depth_to_heightmap.py "$@"
    ;;
  mujoco-contract)
    exec python scripts/export_mujoco_contract.py --headless "$@"
    ;;
  replay)
    tutorial="${PMT_TUTORIAL_DIR:-$HOME/projects/tutorial}"
    bash "$tutorial/isaacsim_gui.sh" display --no-hints
    exec python scripts/replay_states.py --display ":${DISPLAY_NUM:-99}" "$@"
    ;;
  smoke)
    exec bash scripts/smoke_test.sh "$@"
    ;;
  *)
    echo "Usage: bash scripts/pmt.sh {train|play|gui-train|gui-play|replay|mujoco-live|mujoco-stop|mujoco-eval|mujoco-contract|depth-replay|test|smoke} [arguments]"
    echo "Example: bash scripts/pmt.sh train --task PMT-G1-MultiMotionV2-Flat-v0 --num_envs 8 --max_iterations 5 --headless"
    ;;
esac
