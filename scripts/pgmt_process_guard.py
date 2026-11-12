"""Monitor native PhysX failures outside the simulator's Python/GIL context."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from scripts.pgmt_distributed import prepare_launch, RankLayout


def physics_failure(line):
    return ('[omni.physx' in line and ('[Error]' in line or 'overflow' in line.lower()))


def run_guarded(command, root):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--log_dir')
    options, _ = parser.parse_known_args(command[2:])
    if options.log_dir is None:
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            parser.error('External torchrun requires an explicit shared --log_dir')
        options.log_dir = str(root / 'logs/pgmt_p245' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        command = [*command, '--log_dir', options.log_dir]
    output = Path(options.log_dir).resolve()
    layout = RankLayout.from_environment(os.environ)
    status_path = layout.output_dir(output) / 'training_status.json'
    if status_path.exists():
        parser.error(f'Refusing to overwrite an existing run: {status_path.parent}')
    run_id = uuid.uuid4().hex
    environment = dict(os.environ, PMT_PGMT_PHYSICS_WORKER='1', PYTHONUNBUFFERED='1', PMT_PGMT_RUN_ID=run_id)
    command, environment = prepare_launch(command, environment)
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    def status_files():
        return list(dict.fromkeys([status_path, *output.glob('ranks/rank_*/training_status.json')]))
    def owned_workers():
        # torchrun starts workers in separate sessions. Do not leave a stuck
        # native PhysX worker alive after terminating the torchrun coordinator.
        pids = []
        for path in status_files():
            try:
                state = json.loads(path.read_text())
                pid = state['pid']
                marker = f'PMT_PGMT_RUN_ID={run_id}'.encode()
                if state.get('run_id') == run_id and marker in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
                    pids.append(pid)
            except (OSError, ValueError, KeyError):
                continue
        return pids
    def stop():
        workers = owned_workers()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for pid in workers:
            try:
                if pid in owned_workers():
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    previous_handlers = {}
    def forward_signal(signum, frame):
        workers = owned_workers()
        if workers:
            for pid in workers:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
        elif process.poll() is None:
            os.killpg(process.pid, signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, forward_signal)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            if physics_failure(line):
                stop()
                # Do not alter a prior run when the worker refused its directory.
                for path in status_files():
                    if not path.is_file():
                        continue
                    state = json.loads(path.read_text())
                    if state.get('pid') == process.pid or state.get('run_id') == run_id:
                        state.update(status='failed_physics', physics_errors=[line.strip()],
                            error='Native PhysX failure; owned worker terminated; results after failure are invalid',
                            updated_at=datetime.now().isoformat())
                        temporary = path.with_suffix('.tmp')
                        temporary.write_text(json.dumps(state, indent=2)+'\n')
                        temporary.replace(path)
                return 1
        code = process.wait()
        if code:
            for path in status_files():
                if not path.is_file():
                    continue
                state = json.loads(path.read_text())
                if state.get('run_id') == run_id and not state.get('status', '').startswith('failed'):
                    state.update(status='failed', error=f'Training worker group exited with code {code}',
                                 updated_at=datetime.now().isoformat())
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(state, indent=2) + '\n')
                    temporary.replace(path)
        return code
    except BaseException:
        stop()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        process.stdout.close()
