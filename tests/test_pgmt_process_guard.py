"""Native physics errors must not become successful PPO runs with stale tensors."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pgmt_process_guard',ROOT/'scripts/pgmt_process_guard.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_physics_failure_filter():
    assert guard.physics_failure('[Error] [omni.physx.plugin] failed to allocate memory')
    assert guard.physics_failure('[Warning] [omni.physx.plugin] collision buffer overflow')
    assert not guard.physics_failure('[Warning] [omni.fabric.plugin] width is not valid')
    assert not guard.physics_failure('[Info] [omni.physx.plugin] initialized')


def test_native_failure_terminates_owned_worker_and_marks_run(tmp_path):
    worker=tmp_path/'worker.py'
    worker.write_text('''import os, json, time, sys
from pathlib import Path
path=Path(sys.argv[sys.argv.index('--log_dir')+1])
path.mkdir()
(path/'training_status.json').write_text(json.dumps(dict(status='training',pid=os.getpid())))
print('[Error] [omni.physx.plugin] failed to allocate memory',flush=True)
time.sleep(30)
(path/'bad_checkpoint').touch()
''')
    run=tmp_path/'run'
    start=time.monotonic()
    result=guard.run_guarded([sys.executable,str(worker),'--log_dir',str(run)],tmp_path)
    assert result==1 and time.monotonic()-start < 10
    assert json.loads((run/'training_status.json').read_text())['status']=='failed_physics'
    assert not (run/'bad_checkpoint').exists()


def test_normal_worker_exit_preserved(tmp_path):
    worker=tmp_path/'worker.py'
    worker.write_text("print('[Info] [omni.physx.plugin] healthy')\n")
    assert guard.run_guarded([sys.executable,str(worker),'--log_dir',str(tmp_path/'unused')],tmp_path)==0


def test_physics_failure_kills_detached_rank_and_marks_entire_run(tmp_path):
    worker = tmp_path / 'coordinator.py'
    worker.write_text('''import os, json, sys, subprocess, time, signal
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--log_dir') + 1])
rank = int('--child' in sys.argv)
path = root / 'ranks/rank_1' if rank else root
path.mkdir(parents=True, exist_ok=True)
(path / 'training_status.json').write_text(json.dumps(dict(
    status='training', pid=os.getpid(), run_id=os.environ['PMT_PGMT_RUN_ID'])))
if rank:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('[Error] [omni.physx.plugin] failed to allocate memory', flush=True)
    time.sleep(30)
    (root / 'bad_checkpoint').touch()
else:
    subprocess.Popen([sys.executable, __file__, '--log_dir', str(root), '--child'], start_new_session=True)
    time.sleep(30)
''')
    run = tmp_path / 'run'
    result = guard.run_guarded([sys.executable, str(worker), '--log_dir', str(run)], tmp_path)
    assert result == 1
    for path in [run, run / 'ranks/rank_1']:
        state = json.loads((path / 'training_status.json').read_text())
        assert state['status'] == 'failed_physics'
        # A killed detached child may briefly remain a zombie awaiting reaping.
        proc = Path(f"/proc/{state['pid']}/stat")
        deadline = time.monotonic() + 2
        while proc.exists() and proc.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
            time.sleep(.01)
        if proc.exists():
            assert proc.read_text().split()[2] == 'Z'
    assert not (run / 'bad_checkpoint').exists()
