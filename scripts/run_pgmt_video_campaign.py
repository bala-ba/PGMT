"""Durable, serial GUI playback and video export using the verified Stage2 entry."""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from pgmt_force_analysis import load_run
from pgmt_reference_replay import load_recorded_reference

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/lzp/miniconda3/envs/pmt/bin/python'
CASES = [('01_flat', 'flat', 'up', '平地'), ('02_rough', 'rough', 'up', '粗糙地面'),
         ('03_up_stair', 'stairs', 'up', '上楼梯'), ('04_down_stair', 'stairs', 'down', '下楼梯'),
         ('05_up_slope', 'slope', 'up', '上坡'), ('06_down_slope', 'slope', 'down', '下坡'),
         ('07_boxes', 'boxes', 'up', '箱体')]
SELECTION = [
    ('walk1_subject1_clip1', '步行 1 / 已核验基线'),
    ('walk2_subject1_clip2', '步行 2'), ('walk3_subject1_clip1', '步行 3'),
    ('walk4_subject1_clip2', '步行 4'), ('run1_subject2_clip1', '跑步 1'),
    ('run2_subject1_clip0', '跑步 2 / 起步片段'), ('run2_subject1_clip1', '跑步 2 / 后续片段'),
    ('sprint1_subject2_clip0', '冲刺'), ('jumps1_subject1_clip0', '跳跃'),
    ('dance1_subject1_clip2', '舞蹈 1'), ('dance2_subject1_clip2', '舞蹈 2'),
    ('fight1_subject2_clip0', '格斗'), ('fightAndSports1_subject1_clip2', '格斗与运动'),
    ('fallAndGetUp1_subject1_clip0', '跌倒起身 1'),
    ('fallAndGetUp2_subject2_clip0', '跌倒起身 2'),
    ('fallAndGetUp3_subject1_clip0', '跌倒起身 3'),
]


def now():
    return datetime.now().isoformat(timespec='seconds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def plan(output, levels, motion_names=None, case_names=None):
    checkpoint = ROOT / 'logs/pgmt_stage2_center2x2/train_03/model_19999.pt'
    motions = []
    selected = SELECTION if motion_names is None else [(name, dict(SELECTION).get(name, '动作')) for name in motion_names]
    cases = CASES if case_names is None else [case for case in CASES if case[0] in case_names]
    for name, label in selected:
        path = ROOT / 'assets/motions/pgmt_clipped_50hz_tcrs_v1' / (name + '.npz')
        with np.load(path) as d:
            xy = d['body_pos_w'][:, 0, :2]
            motions.append(dict(name=name, label=label, path=str(path), sha256=digest(path),
                frames=len(xy), source_fps=float(d['fps'][0]), initial_frame=0,
                initial_contacts=d['contacts'][0].tolist(),
                initial_max_joint_speed=float(abs(d['joint_vel'][0]).max()),
                path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())))
    trials = []
    for mi, motion in enumerate(motions):
        for level in levels:
            for case, terrain, direction, label in cases:
                trials.append(dict(id=f"level{level}/{motion['name']}/{case}", motion_index=mi,
                    motion=motion['name'], level=level, case=case, terrain=terrain,
                    direction=direction, terrain_label=label, state='pending'))
    return dict(created_at=now(), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                levels=levels, motions=motions, trials=trials, cases=cases, initial_frame=0, patch_id=5,
                seed=1017, speed=1, display=':1', device='cuda:1',
                simulator='Isaac / PhysX; original GUI entry',
                training=False, reference_enabled=True, perception_enabled=True)


def command_for(p, row, directory):
    return ['bash', 'scripts/play_pgmt_stage2_stairs.sh', '--checkpoint', p['checkpoint'],
        '--motion', p['motions'][row['motion_index']]['path'], '--terrain', row['terrain'],
        '--level', str(row['level']), '--direction', row['direction'], '--patch_id', '5',
        '--speed', '1', '--max_episodes', '0', '--record_perception', '--log_dir', str(directory)]


def terminate_owned(process, status_path):
    # The existing process guard forwards SIGTERM to its own native worker.
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            if status_path and status_path.exists():
                try:
                    state = json.loads(status_path.read_text())
                    pid, run_id = state['pid'], state['run_id']
                    marker = f'PMT_PGMT_RUN_ID={run_id}'.encode()
                    if marker in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') and os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGKILL)
                except (OSError, KeyError, ValueError):
                    pass
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def launch(command, log, *, status_path=None, timeout=360):
    environment = dict(os.environ, DISPLAY=':1', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='1')
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            terminate_owned(process, status_path)
            raise
    if result:
        raise RuntimeError(f'Process exited {result}; see {log}')


def audit(directory, p, row):
    run = load_run(directory)
    meta, data = run['metadata'], run['data']
    assert meta['completed_episodes'] == 1
    assert meta['source_sha256'] == p['checkpoint_sha256']
    assert meta['policy_state_unchanged'] and meta['source_checkpoint_unchanged']
    assert meta['force_recording']['motion_sha256'] == p['motions'][row['motion_index']]['sha256']
    assert meta['force_recording']['physics_buffer_capacities'] == {}
    for key, value in dict(level=row['level'], terrain=row['terrain'], direction=row['direction'],
                           patch_id=5, speed=1, max_episodes=0, headless=False, device='cuda:1').items():
        assert meta['arguments'][key] == value, key
    cfg = yaml.load((directory / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
    baseline = yaml.load((ROOT / 'logs/pgmt_terrain_suite/20261104_train03_17500_walk_slope_l1_frame0/level1/walk1_subject1_clip1/05_up_slope/env.yaml').read_text(), Loader=yaml.BaseLoader)
    for key in ('sim', 'events', 'terminations', 'observations'):
        assert cfg[key] == baseline[key], key
    assert cfg['commands']['motion']['fixed_start_frame'] == '0'
    assert cfg['commands']['motion']['require_valid_support_start'] == 'false'
    with np.load(directory / 'perception.npz') as d:
        perception = dict(d)
    n, dec = meta['steps'], meta['force_recording']['decimation']
    assert len(perception['time_s']) == n + 1 and len(data['time_s']) == n * dec
    assert perception['frame'][0] == 0 and perception['time_s'][0] == 0
    np.testing.assert_array_equal(data['frame'], np.repeat(np.arange(n), dec))
    np.testing.assert_allclose(data['time_s'], np.arange(1, n * dec + 1) * meta['physics_dt'], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(perception['body_pos_w'][-1], data['body_pos_w'][-1])
    assert data['done'][-dec:].all() and not data['done'][:-dec].any()
    for key in ('body_pos_w', 'body_quat_w', 'joint_pos', 'terrain_map', 'attention_weights'):
        assert np.isfinite(perception[key]).all(), key
    reference, proof = load_recorded_reference(run, perception)
    np.testing.assert_array_equal(perception['joint_pos'][0], reference['reference_joint_pos'][0])
    error = np.linalg.norm(perception['body_pos_w'][:, 0, :2] - reference['reference_body_pos_w'][:, 0, :2], axis=-1)
    assert error[0] < 1e-4
    ep = meta['last_episode']
    result = dict(seconds=ep['seconds'], failed=ep['failed'], motion_finished=ep['motion_finished'],
        termination_flags=ep['termination_flags'], initial_frame=0, initial_root_error_m=float(error[0]),
        root_error_1s_m=float(error[np.argmin(abs(perception['time_s'] - 1))]) if ep['seconds'] >= 1 else None,
        root_error_max_m=float(error.max()), physics_samples=n * dec,
        source_checkpoint_unchanged=True, reference_validation=proof)
    write_json(directory.parent / 'audit.json', result)
    return result


def publish(output, p, state):
    rows = state['trials']
    ready = [r for r in rows if r['state'] == 'ready']
    failed = [r for r in rows if r['state'] == 'error']
    state.update(updated_at=now(), videos_ready=len(ready), total_trials=len(rows),
                 tracking_completed=sum(not r['result']['failed'] and r['result']['motion_finished'] for r in ready),
                 tracking_failed=sum(r['result']['failed'] for r in ready), infrastructure_errors=len(failed))
    write_json(output / 'progress.json', state)
    write_json(output / 'summary.json', {k: v for k, v in state.items() if k != 'trials'})
    with (output / 'results.csv').open('w', newline='') as f:
        fields = ['level', 'motion', 'case', 'state', 'seconds', 'failed', 'motion_finished', 'root_error_1s_m', 'video', 'error']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row.get('result', {})})
    e = html.escape
    cases = p.get('cases', CASES)
    levels_text = ' / '.join(str(level) for level in p['levels'])
    blocks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PGMT 仿真测试</title>',
        '<style>body{font:16px system-ui;background:#111b27;color:#e4edf5;margin:32px}a{color:#7ddbe9}table{border-collapse:collapse;width:100%;margin:16px 0 36px}td,th{border:1px solid #3a4b5a;padding:12px;text-align:left}small{display:block;color:#a7bbc9;margin-top:8px}.ok{color:#8de3ac}.fail{color:#ffab91}code{font-size:13px}video{max-width:720px;width:100%;background:#000}h2{margin-top:40px}td{min-width:90px}</style>',
        f'<h1>model_19999 · Level {levels_text} · {len(p["motions"])} 段动作</h1><p>视频 {len(ready)}/{len(rows)}；动作完成 {state["tracking_completed"]}；提前终止 {state["tracking_failed"]}；运行错误 {len(failed)}。</p>',
        f'<p>任务状态：{e(state["status"])} · 更新：{e(state["updated_at"])}（北京时间；刷新页面查看进度）</p>',
        '<p>原始第 0 帧起步 · patch 5 · seed 1017 · 1× 播放 · 原 GUI 仿真配置。灰色为实际机器人，青蓝色为参考动作（展示横移 1 m）。保留策略高度扫描、感知块与注意力。每个组合单独成片，提前终止结果保留。</p>',
        '<p>下行从地形边缘开始；完整动作可能转向或进入相邻地形。舞蹈、格斗及跌倒起身作为扩展动作测试，命名按源动作类别。每组只运行一次，完成数不等同于统计成功率。</p>',
        '<p><a href="results.csv">结果表</a> · <a href="progress.json">详细进度</a> · <a href="selection.json">动作与配置清单</a></p>',
        '<h2 id="player-title">选择下方视频播放</h2><video id="player" controls preload="none"></video>']
    for level in p['levels']:
        blocks.append(f'<h2>Level {level}</h2><table><tr><th>动作</th>' + ''.join(f'<th>{label}</th>' for _, _, _, label in cases) + '</tr>')
        for motion in p['motions']:
            blocks.append(f'<tr><td>{e(motion["label"])}<small><code>{e(motion["name"])}</code></small></td>')
            for case, _, _, _ in cases:
                row = next(r for r in rows if r['level'] == level and r['motion'] == motion['name'] and r['case'] == case)
                if row['state'] == 'ready':
                    result = row['result']
                    text = '提前终止' if result['failed'] else ('动作完成' if result['motion_finished'] else '结束')
                    title = f'Level {level} / {motion["label"]} / {row["terrain_label"]}'
                    blocks.append(f'<td><a class="play" data-title="{e(title)}" href="{e(row["video"])}">播放视频</a><small class="{"fail" if result["failed"] else "ok"}">{text} {result["seconds"]:.2f}s</small><small><a href="{e(row["video"])}" target="_blank">单独打开</a></small></td>')
                else:
                    message = {'pending':'待测试', 'simulating':'仿真中', 'rendering':'导出中', 'error':'运行错误'}[row['state']]
                    blocks.append(f'<td title="{e(row.get("error", ""))}">{message}</td>')
            blocks.append('</tr>')
        blocks.append('</table>')
    blocks.append('<script>document.querySelectorAll("a.play").forEach(a=>a.onclick=e=>{e.preventDefault();let v=document.getElementById("player");v.src=a.href;document.getElementById("player-title").textContent=a.dataset.title;v.scrollIntoView({behavior:"smooth"});v.play();});</script></html>')
    tmp = output / 'index.tmp'
    tmp.write_text('\n'.join(blocks))
    tmp.replace(output / 'index.html')


def export(output, p, row, trial, recording):
    # A one-trial manifest keeps each rendered video independently auditable.
    motion = p['motions'][row['motion_index']]
    suite = dict(checkpoint=p['checkpoint'], source_sha256=p['checkpoint_sha256'],
        levels=[row['level']], motions=[motion], trials=[dict(level=row['level'], motion_index=0,
        label=row['case'][3:], terrain=row['terrain'], direction=row['direction'],
        directory=str(recording.relative_to(trial)))])
    write_json(trial / 'suite.json', suite)
    validation = None
    errors = []
    for attempt in range(1, 3):
        render_root = trial / f'render_attempt_{attempt:02d}'
        proof = render_root / motion['name'] / 'validation.json'
        try:
            if not proof.exists():
                launch([PYTHON, 'scripts/render_pgmt_perception_suite.py', '--source', str(trial),
                        '--output', str(render_root), '--show_reference', '--individual_only'],
                       trial / f'render_attempt_{attempt:02d}.log')
            candidate = json.loads(proof.read_text())
            assert digest(candidate['videos'][0]['file']) == candidate['videos'][0]['sha256']
            validation = candidate
            break
        except Exception as error:
            errors.append(repr(error))
    if validation is None:
        raise RuntimeError('Video export failed: ' + '; '.join(errors))
    video = Path(validation['videos'][0]['file'])
    assert digest(video) == validation['videos'][0]['sha256']
    target = output / 'videos' / f"level{row['level']}" / motion['name'] / video.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert digest(target) == digest(video), 'Conflicting prior video'
    else:
        os.link(video, target)
    row['video'] = str(target.relative_to(output))
    write_json(target.with_suffix('.validation.json'), validation)
    row['validation'] = str(target.with_suffix('.validation.json').relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--motions', nargs='+', help='Motion basenames without .npz; default: representative campaign selection')
    parser.add_argument('--cases', nargs='+', choices=[case[0] for case in CASES], help='Optional terrain/direction subset')
    parser.add_argument('--prepare_only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = output / 'selection.json'
    if selection.exists():
        p = json.loads(selection.read_text())
    else:
        p = plan(output, args.levels, args.motions, args.cases)
        write_json(selection, p)
    path = output / 'progress.json'
    state = json.loads(path.read_text()) if path.exists() else dict(status='prepared', trials=p['trials'])
    if args.prepare_only:
        publish(output, p, state)
        print(f"Prepared {len(p['trials'])} trials at {output}")
        return
    import fcntl
    lock = (output / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state.update(status='running', pid=os.getpid(), started_at=state.get('started_at', now()))
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Campaign received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    publish(output, p, state)
    for row in state['trials']:
        if row['state'] in ('ready', 'error'):
            continue
        if digest(p['checkpoint']) != p['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during campaign')
        trial = output / 'trials' / row['id']
        trial.mkdir(parents=True, exist_ok=True)
        try:
            recording = None
            errors = []
            for attempt in range(1, 3):
                candidate = trial / f'attempt_{attempt:02d}' / row['case']
                candidate.parent.mkdir(parents=True, exist_ok=True)
                row.update(state='simulating', started_at=now(), attempt=attempt)
                publish(output, p, state)
                command = command_for(p, row, candidate)
                write_json(candidate.parent / 'command.json', dict(argv=command, cwd=str(ROOT), DISPLAY=':1'))
                try:
                    if not (candidate / 'training_status.json').exists():
                        launch(command, candidate.parent / 'simulation.log', status_path=candidate / 'training_status.json')
                    row['result'] = audit(candidate, p, row)
                    recording = candidate
                    break
                except Exception as error:
                    errors.append(repr(error))
                    row['attempt_errors'] = errors
                    print(f"[RETRY] {row['id']} attempt {attempt}: {error}", flush=True)
            if recording is None:
                raise RuntimeError('; '.join(errors))
            row.update(state='rendering', recording=str(recording.relative_to(output)))
            publish(output, p, state)
            export(output, p, row, trial, recording)
            row.update(state='ready', finished_at=now())
            print(f"[READY] {row['id']} {row['result']['seconds']:.2f}s failed={row['result']['failed']}", flush=True)
        except Exception as error:
            row.update(state='error', error=repr(error), finished_at=now())
            print(f"[ERROR] {row['id']}: {error}", flush=True)
        publish(output, p, state)
    state.update(status='completed_with_errors' if any(r['state'] == 'error' for r in state['trials']) else 'completed', finished_at=now())
    publish(output, p, state)
    print(json.dumps({k: v for k, v in state.items() if k != 'trials'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
            if (output / 'progress.json').exists() and (output / 'selection.json').exists():
                state = json.loads((output / 'progress.json').read_text())
                p = json.loads((output / 'selection.json').read_text())
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'campaign_error',
                             error=repr(error), finished_at=now())
                publish(output, p, state)
        raise
