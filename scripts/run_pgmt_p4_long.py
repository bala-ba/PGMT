"""Durable single-GPU PGMT run with explicit OOM fallback and audit journal.

Run after finite P4 gates pass. Each fallback is a fresh, separately named run
from the approved Stage1 checkpoint; never pretend different env counts can
strictly resume an old physical curriculum state.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def is_oom(state, log):
    text = (json.dumps(state) + log).lower()
    return any(term in text for term in ("out of memory", "outofmemoryerror", "failed to allocate",
                                        "cuda_error_out_of_memory", "cudaerrormemoryallocation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--hours", type=float, default=12.)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--num_envs", type=int, nargs="+", default=[8192, 4096, 2048, 1024])
    parser.add_argument("--checkpoint", default=str(ROOT/"logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"))
    args = parser.parse_args()
    if args.hours < 0 or args.max_iterations < 1 or min(args.num_envs) < 5:
        parser.error("Invalid training budget")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for gate in (root / "finite_acceptance.json", root / "physics_01/training_status.json"):
        if read_json(gate).get("status") != "passed":
            parser.error(f"Finite P4 acceptance must pass first: {gate}")
    status_path = root / "overnight_status.json"
    if status_path.exists():
        parser.error("Refusing to overwrite an existing overnight supervisor")
    stop_file = root / "STOP_REQUESTED"
    journal = root / "EXECUTION.md"
    state = dict(status="starting", supervisor_pid=os.getpid(), device="cuda:0",
                 requested_num_envs=args.num_envs, started_at=datetime.now().isoformat(),
                 hours=args.hours, max_iterations=args.max_iterations, attempts=[])
    started = time.monotonic()

    def note(message):
        with journal.open("a") as f:
            f.write(f"\n- {datetime.now().isoformat(timespec='seconds')}：{message}\n")
        print(message, flush=True)

    def stop(signum, frame):
        stop_file.touch()
        note(f"监督进程收到 {signal.Signals(signum).name}，请求在更新边界保存退出。")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    atomic_json(status_path, state)
    for count in args.num_envs:
        remaining = max(1., args.hours*3600 - (time.monotonic()-started)) if args.hours else 0.
        if stop_file.exists() or (args.hours and remaining <= 1.):
            state.update(status="stopped_before_next_attempt")
            break
        output = root / f"formal_{count}"
        output.mkdir(exist_ok=False)
        command = ["bash", "scripts/train_pgmt_terrain.sh", "--headless", "--device", "cuda:0",
                   "--num_envs", str(count), "--max_iterations", str(args.max_iterations),
                   "--save_interval", "50", "--max_runtime_s", str(remaining), "--stop_file", str(stop_file),
                   "--initialize-from", str(Path(args.checkpoint).resolve()), "--log_dir", str(output)]
        atomic_json(output/"supervisor_launch.json", dict(command=command, note="Fresh Stage1 initialization for this environment count"))
        with (output/"console.log").open("w") as console:
            process = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            attempt = dict(num_envs=count, output=str(output), guard_pid=process.pid, started_at=datetime.now().isoformat())
            state["attempts"].append(attempt)
            state.update(status="running", active_run=str(output), active_num_envs=count)
            atomic_json(status_path, state)
            note(f"开始正式训练尝试：cuda:0，{count} 环境，输出 `{output}`。")
            last_noted, stop_started = -1, None
            while process.poll() is None:
                child = read_json(output/"training_status.json")
                state.update(worker_pid=child.get("pid"), training_status=child.get("status"),
                             completed_updates=child.get("completed_updates", 0),
                             next_iteration=child.get("next_iteration"), updated_at=datetime.now().isoformat())
                update = state["completed_updates"]
                if update > 0 and (last_noted < 0 or update - last_noted >= 100):
                    latest = child.get("latest", {})
                    note(f"{count} 环境已完成 {update} 次更新；KL={latest.get('losses', {}).get('approx_kl')}，"
                         f"LR={latest.get('learning_rate')}；检查点与完整指标在运行目录。")
                    last_noted = update
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader"], capture_output=True, text=True)
                with (root/"gpu_monitor.jsonl").open("a") as f:
                    f.write(json.dumps(dict(time=datetime.now().isoformat(), num_envs=count,
                                            updates=update, gpu=gpu.stdout.strip()))+"\n")
                if args.hours and time.monotonic()-started >= args.hours*3600:
                    stop_file.touch()
                if stop_file.exists():
                    if stop_started is None:
                        stop_started = time.monotonic()
                        note("收到停止请求/达到时间预算，等待当前 PPO 更新完成并保存 final.pt。")
                    if time.monotonic()-stop_started > 600:
                        # An unresponsive native worker must not run past the
                        # budget indefinitely. Kill only this run's owned groups.
                        for pid in (child.get("pid"), process.pid):
                            if pid:
                                try:
                                    os.killpg(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                        state["forced_stop"] = "Worker unresponsive for 600s after stop request; use last periodic checkpoint"
                        note(state["forced_stop"])
                atomic_json(status_path, state)
                time.sleep(30)
            process.wait()
        child = read_json(output/"training_status.json")
        attempt.update(exit_code=process.returncode, status=child.get("status"),
                       completed_updates=child.get("completed_updates", 0), finished_at=datetime.now().isoformat())
        if process.returncode == 0 and child.get("status") == "completed":
            state.update(status="completed", final_checkpoint=child.get("final_checkpoint"),
                         completed_updates=child.get("completed_updates", 0), next_iteration=child.get("next_iteration"),
                         stop_reason=child.get("stop_reason"))
            note(f"正式训练正常结束：{count} 环境，{state['completed_updates']} 次更新；`{state['final_checkpoint']}`。")
            break
        log_tail = (output/"console.log").read_text(errors="replace")[-100000:]
        if stop_file.exists() or state.get("forced_stop"):
            state.update(status="stopped_without_final_checkpoint")
            break
        if is_oom(child, log_tail):
            attempt["fallback_reason"] = "OOM"
            note(f"{count} 环境因 OOM 退出；保留该运行全部材料，下一档从指定 Stage1 权重重新初始化。")
            atomic_json(status_path, state)
            continue
        state.update(status="failed", error=child.get("error", f"exit {process.returncode}"))
        note(f"非 OOM 错误，停止自动重试：{state['error']}。")
        break
    else:
        state.update(status="all_sizes_oom")
    state["finished_at"] = datetime.now().isoformat()
    atomic_json(status_path, state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
