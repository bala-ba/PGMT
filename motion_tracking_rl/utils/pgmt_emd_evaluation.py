"""Isolated native Isaac EMD evaluation at PPO rollout boundaries."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time

import torch


class PGMTEMDEvaluation:
    def __init__(self, runner, command):
        self.runner, self.command = runner, command

    def after_update(self, completed_iterations, *, force=False):
        runner, sampler = self.runner, self.command.sampler
        global_step = (completed_iterations * runner.num_steps_per_env * runner.env.num_envs
                       * runner.gpu_world_size)
        if not force and not sampler.due(global_step):
            return
        started = time.monotonic()
        result = None
        if runner.gpu_global_rank == 0:
            try:
                result = dict(scores=self._evaluate(global_step))
            except Exception as error:
                # Broadcast errors as well: all ranks must fail together.
                result = dict(error=f"{type(error).__name__}: {error}")
        if runner.is_distributed:
            payload = [result]
            torch.distributed.broadcast_object_list(payload, src=0)
            result = payload[0]
        if "error" in result:
            raise RuntimeError(f"PGMT EMD evaluation failed; priorities were not changed: {result['error']}")
        sampler.update_emd(result["scores"], global_step)
        if runner.writer is not None and not runner.disable_logs:
            runner.writer.add_scalar("EMD/evaluation_seconds", time.monotonic() - started, completed_iterations)
            runner.writer.add_scalar("EMD/mean", float(sampler.emd.mean()), completed_iterations)
            runner.writer.add_histogram("EMD/per_motion", sampler.emd.cpu(), completed_iterations)
        print(f"[PGMT EMD] step={global_step}, clips={sampler.num_motions}, "
              f"mean={float(sampler.emd.mean()):.6f}, updated all clip priorities", flush=True)

    def _evaluate(self, global_step):
        runner, command = self.runner, self.command
        raw = runner.env.unwrapped
        root = Path(__file__).resolve().parents[2]
        parent = Path(runner.log_dir or tempfile.gettempdir()) / "emd_evaluations"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"step_{global_step}_", dir=parent))
        snapshot = output / "policy.pt"
        metadata = runner.alg.policy.get_checkpoint_metadata()
        files = [str(Path(m.source_file).resolve()) for m in command.data_store.motions]
        request = dict(model_state_dict={k: v.detach().cpu() for k, v in runner.alg.policy.state_dict().items()},
                       policy_metadata=metadata, motion_files=files, global_step=global_step,
                       sampling_config=command.sampler.config,
                       decimation=raw.cfg.decimation, sim_dt=raw.cfg.sim.dt,
                       robot_usd=getattr(raw.cfg.scene.robot.spawn, "usd_path", None))
        torch.save(request, snapshot)
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        for key in list(environment):
            if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                       "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") or key.startswith("TORCHELASTIC_"):
                environment.pop(key)
        args = [sys.executable, str(root / "scripts/eval_pgmt_emd.py"), "--request", str(snapshot),
                "--output", str(output / "result.json"), "--headless", "--device", str(runner.device)]
        # Keep the parent's supported Kit experience (the local launcher uses
        # legacy Isaac Lab on Sim 6). These are runtime options, not task data.
        for option in ("--experience", "--kit_args"):
            for i, argument in enumerate(sys.argv):
                if argument == option and i + 1 < len(sys.argv):
                    args.extend([option, sys.argv[i + 1]])
                elif argument.startswith(option + "="):
                    args.append(argument)
        print(f"[PGMT EMD] full-library flat evaluation: {output}", flush=True)
        log_path = output / "worker.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(args, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"[PGMT EMD] evaluating, elapsed={time.monotonic()-started:.0f}s; {log_path}", flush=True)
                if process.returncode:
                    raise RuntimeError(f"Worker exit {process.returncode}; see {log_path}")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        log_text = log_path.read_text(errors="replace")
        from scripts.pgmt_process_guard import physics_failure
        if any(physics_failure(line) for line in log_text.splitlines()):
            raise RuntimeError(f"PhysX reported an error during EMD evaluation; see {log_path}")
        result = json.loads((output / "result.json").read_text())
        if (result.get("motion_files") != files or result.get("global_step") != global_step
                or result.get("metric") != "joint_uniform_emd_euclidean"):
            raise ValueError("EMD worker returned a different motion library/metric/step")
        # The durable result includes checkpoint hash and scores. Avoid retaining
        # another complete policy copy every evaluation period.
        snapshot.unlink()
        return result["emd"]
