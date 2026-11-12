# PMT server environment and visualization

Run commands from the PMT repository root. The isolated conda environment is
`pmt`; `isaacsim`, `mimiclite-mjlab`, and other existing environments are retained.
See [the completed acceptance report](SMOKE_REPORT_2026-09-23.md) for measured
results, saved checkpoints, training curves, and videos.

## Environment

The runtime uses Python 3.11, Isaac Lab 2.3.2 (official pip wheel), Isaac Sim
5.1.0, and PyTorch 2.7.0 with CUDA 12.8. This CUDA build supports the server's
RTX 5090 GPUs. The version family follows the
[Isaac Lab installation instructions](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html).

Recreate from the repository root:

```bash
conda env create -f environment.yml
conda activate pmt
python scripts/download_robot_assets.py
python -m pip check
source scripts/pmt_env.sh
```

`requirements-pmt.txt` pins the project runtime. In particular, Isaac Sim 5.1
requires NumPy 1.26.0, packaging 23.0, matplotlib 3.10.3, trimesh 4.5.1,
typing-extensions 4.12.2, and Pillow 11.3.0. TensorDict 0.9.0 avoids the
`pyvers` / `packaging>=25` conflict introduced by TensorDict 0.9.1. Installing
the official Isaac Lab wheel also avoids dependency changes in the separate
local `IsaacLab-2.3.2` checkout.

`scripts/pmt_env.sh` locates the downloaded G1 robot and the bundled motion and
terrain assets. Existing `PMT_*` overrides are respected. The default flat
smoke data is the bundled `walk_dance1sub2start/raw` directory, not the original
training dataset. The perceptive task's original subject-1 data is not bundled;
pass `--motion_file` to use the included subject-2 demonstration data.

## Training and smoke tests

The launcher activates conda and configures paths. `GPU=0` is the default;
override it with `GPU=1` or an explicit `CUDA_VISIBLE_DEVICES` value.

```bash
bash scripts/pmt.sh smoke

bash scripts/pmt.sh train --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 8 --max_iterations 5 --seed 42 --headless

bash scripts/pmt.sh play --task PMT-G1-MultiMotionV2-Flat-v0 \
  --resume_path checkpoints/pretrained/multimotionv2_flat.pt \
  --num_envs 2 --headless --max_steps 300 --metrics_file logs/play/metrics.json
```

Smoke tests check actual gradient updates, checkpoint saving/reloading and
finite rollout tensors. A handful of PPO iterations does **not** establish
convergence or tracking quality. Pretrained checkpoints and newly trained
smoke checkpoints are evaluated separately.

The suite covers flat PPO, the perceptive token tracker, real-teacher vision
distillation, and PPO finetuning of the newly distilled student. Distillation
uses the bundled teacher via `network.teacher_ckpt=...`; finetuning uses
`network.base_policy_ckpt=...`. These explicit overrides replace paths to
historical training runs that are not present on this server.

Training writes checkpoints, TensorBoard events and resolved `params/env.yaml`
and `params/agent.yaml`. Playback writes a JSON summary when `--metrics_file`
is supplied. Both entrypoints accept supported task-config `key=value`
overrides and `--motion_file`; paths are applied before constructing the task.

## VNC visualization (projects/tutorial)

The GUI launcher reuses `~/projects/tutorial/isaacsim_gui.sh display --no-hints`
and the Mesa/MuJoCo approach in `~/projects/tutorial/MUJOCO_GUI.md`:
Isaac Lab states → MuJoCo viewer → Xvfb `:99` → localhost x11vnc `5900` → Tailscale.
The simulator and renderer both come from the new **pmt** environment.
The helper's `run` command uses its hard-coded `isaacsim` environment and should
not be used for PMT.

```bash
bash scripts/pmt.sh gui-train --task PMT-G1-MultiMotionV2-Flat-v0 \
  --num_envs 4 --max_iterations 12 --seed 42 \
  --video --video_length 288 --log_dir logs/visualization/my_flat_train

bash scripts/pmt.sh gui-play --task PMT-WalkDanceBigMap-G1-v0 \
  --resume_path checkpoints/pretrained/walkdance_bigmap_teacher.pt \
  --num_envs 1 --max_steps 500 --real_time \
  --video --video_length 500 --video_dir logs/visualization/my_terrain_play \
  --metrics_file logs/visualization/my_terrain_play/metrics.json
```

Connect from a computer on the existing Tailscale network to the VNC address
printed by the launcher. The existing VNC password is documented in
`~/projects/tutorial/GUI.md`. At setup time the address was
`vnc://100.122.53.37:5900`; use the tutorial `status` command to verify it.

On this server, native Isaac Sim 5.1 RTX rendering fails with a Vulkan
presentation error / `librtx.scenedb.plugin.so` crash, including offscreen
attempts. Headless PhysX simulation works. The GUI launcher therefore adds
`--headless --state_viz --state_viz_display :99`. This is a display bridge:
Isaac Lab still computes physics, observations, rewards, training and inference.
The separate MuJoCo process only calls `mj_forward` to draw actual robot-0 root
and joint positions; it never steps its own physics. The actual terrain mesh is
also displayed. Joint names and wxyz quaternions are mapped explicitly, and each
frame checks corresponding body positions against Isaac Lab (1 mm tolerance).

The separate `mujoco-eval` command now evaluates the paired Bigmap PPOft student
with actual CPU MuJoCo physics and ray casting. See [standalone MuJoCo validation](MUJOCO_STUDENT_VALIDATION.md)
for its scope, measured tracking errors, artifacts, and reproduction commands.

The camera follows robot 0. `--video` records the first `--video_length` control
steps at the task's simulation frame rate. Each output directory includes an
MP4, `preview.png`, recorded `states.jsonl`, provenance in `metadata.json`, and
mapping checks in `render_result.json`. The live viewer continues after the
recording limit. `--video_interval` applies only to native Kit recording, not to
this bridge. The synchronous software renderer slows GUI training; use plain
`train --headless` for throughput. Both mechanisms are opt-in.

Ending a bounded run leaves the shared VNC display available. Recordings remain
after the simulator exits. Direct `train/play --video` without `--state_viz`
selects the native Kit renderer and currently fails on this server.

Replay saved states in the VNC viewer without running a GPU simulator:

```bash
bash scripts/pmt.sh replay logs/visualization/terrain_teacher --loops 3
```

The replay keeps the original files and writes its own output under `logs/replay`.
New terrain recordings also preserve ray-hit positions and the exact returned
student height observations. The live viewer, MP4, and replay show a moving 3D
sampling grid plus terrain/policy heatmaps; pink cells indicate invalid or
dropped samples. Blind teachers show terrain only. Older recordings without
scan data continue to work without the overlay. See [height-scan visualization
commands](visualize.md#高程图可视化) for the PPO-finetuned student example.
To check MP4 frame counts, simulation frame rate, nonblank/moving images, and
pose correspondence, use `python scripts/check_recordings.py <video-directory>`.

For training curves:

```bash
conda activate pmt
tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

Use an SSH tunnel: `ssh -L 6006:127.0.0.1:6006 <server>` and browse
`http://localhost:6006`. The command above binds to loopback only. Static plots
are also exported with `python scripts/summarize_smoke.py <smoke-directory>`.
