"""Summarize recorded PGMT reward contributions; no simulator or model updates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=Path('logs/pgmt_stage2_center2x2/train_01'))
parser.add_argument('--window', type=int, default=1000)
args = parser.parse_args()
if args.window < 1:
    parser.error('--window must be positive')
run = args.run.resolve()
out = run / 'analysis/terrain_rewards'
out.mkdir(parents=True, exist_ok=True)
saved = torch.load(run / 'model_19999.pt', map_location='cpu', weights_only=False)
contract = saved['training_contract']
environment = contract['runtime']['environment']
dt = environment['sim']['dt'] * environment['decimation']
transitions = environment['scene']['num_envs'] * contract['num_steps_per_env']
weights = {k: v['weight'] for k, v in environment['rewards'].items() if k.startswith('terrain_')}
terms = list(weights)
families = ['flat', 'stairs', 'slope', 'boxes', 'rough']
columns, rows, iterations, outcomes = None, [], [], []
digest = hashlib.sha256()
max_rate_error = max_terrain_sum_error = max_total_error = 0.
with (run / 'progress.jsonl').open('rb') as stream:
    for i, line in enumerate(stream):
        digest.update(line)
        record = json.loads(line)
        assert record['iteration'] == i and record['completed_iteration'] == i + 1
        t = record['telemetry']
        assert t['Window/transitions'] == transitions
        if columns is None:
            columns = [k for k in t if k.startswith(('RewardStep/', 'RewardGroupStep/', 'TrackingSnapshot/', 'Sampler/'))]
            columns += [f'Curriculum/{f}/mean_level' for f in families]
            columns += [f'Loss/value_function_{g}' for g in environment['pgmt_reward_groups']]
            columns += [f'Gradient/{g}' for g in environment['pgmt_reward_groups']]
        values = dict(t)
        values.update({f'Loss/{k}': v for k, v in record['losses'].items()})
        values.update({f'Gradient/{k}': v for k, v in record['value_head_gradient_l1'].items()})
        rows.append([values[k] for k in columns])
        iterations.append(i)
        counts = [[sum(t[f'OutcomeWindow/{f}/L{level}/{kind}'] for level in range(10))
                   for kind in ('success', 'failure', 'map_exit', 'timeout', 'incomplete_motion', 'other_truncation')]
                  for f in families]
        outcomes.append(counts)
        max_rate_error = max(max_rate_error, max(abs(t[f'RewardStep/{term}'] / dt - t[f'RewardRate/{term}']) for term in terms))
        max_terrain_sum_error = max(max_terrain_sum_error, abs(sum(t[f'RewardStep/{term}'] for term in terms) - t['RewardGroupStep/terrain']))
        max_total_error = max(max_total_error, abs(sum(t[f'RewardGroupStep/{g}'] for g in environment['pgmt_reward_groups']) - t['RewardStep/total']))
        if (i + 1) % 5000 == 0:
            print(f'Read and verified {i+1} iterations', flush=True)

data = np.array(rows)
counts = np.array(outcomes, dtype=np.int64)
assert len(rows) == saved['next_iteration'] == 20000
assert np.isfinite(data).all()
assert max(max_rate_error, max_terrain_sum_error, max_total_error) < 1e-10
index = {k: i for i, k in enumerate(columns)}


def describe(start, end):
    chunk = data[start:end]
    means = {name: float(chunk[:, i].mean()) for i, name in enumerate(columns)}
    terrain = means['RewardGroupStep/terrain']
    positive = sum(means['RewardStep/' + t] for t in terms if weights[t] > 0)
    negative = -sum(means['RewardStep/' + t] for t in terms if weights[t] < 0)
    c = counts[start:end].sum(0)
    result = dict(iteration_start=start, iteration_end=end-1, means=means,
        terrain_positive=positive, terrain_penalty_magnitude=negative, terrain_net=terrain,
        terrain_net_fraction_total=terrain / means['RewardStep/total'],
        terrain_absolute_fraction_all=sum(abs(means['RewardStep/' + t]) for t in terms) /
            sum(abs(v) for k, v in means.items() if k.startswith('RewardStep/') and k != 'RewardStep/total'),
        success_rate=float(c[:, 0].sum()/c.sum()) if c.sum() else None,
        success_by_family={f:float(c[j,0]/c[j].sum()) if c[j].sum() else None for j,f in enumerate(families)},
        ended_episodes=int(c.sum()),
        touchdown_quality_weighted_landings_per_second=means['RewardStep/terrain_touchdown_quality']/(weights['terrain_touchdown_quality']*dt*dt),
        reference_match_rate=means['RewardStep/terrain_reference_contact_match']/(weights['terrain_reference_contact_match']*dt),
        fast_switch_foot_events_per_second=means['RewardStep/terrain_contact_switching']/(weights['terrain_contact_switching']*dt*dt),
        stumble_foot_steps_per_second=means['RewardStep/terrain_stumble']/(weights['terrain_stumble']*dt*dt),
        contact_gated_sum_speed_sq=means['RewardStep/terrain_slip']/(weights['terrain_slip']*dt),
        sum_force_squared_n2=means['RewardStep/terrain_contact_force']/(weights['terrain_contact_force']*dt))
    return result


windows = [describe(i, min(i+args.window, len(rows))) for i in range(0, len(rows), args.window)]
summary = dict(run=str(run), checkpoint=str(run/'model_19999.pt'), iterations=len(rows),
    envs=environment['scene']['num_envs'], steps_per_iteration=contract['num_steps_per_env'], dt=dt,
    samples_per_iteration=transitions, weights=weights, source_progress_sha256=digest.hexdigest(),
    config={k:environment[k] for k in ('pgmt_contact_events','pgmt_touchdown_quality','pgmt_stumble','pgmt_tracking_relaxation')},
    numerical_checks=dict(max_rate_conversion_error=max_rate_error, max_terrain_sum_error=max_terrain_sum_error,
                          max_total_sum_error=max_total_error, all_values_finite=True),
    first_iteration=describe(0,1), first_window=windows[0], last_window=windows[-1],
    whole_run=describe(0,len(rows)), windows=windows,
    note='RewardStep is the signed, already weighted and dt-integrated mean over every environment/control step. Windows are transition-weighted; all iterations have identical sample counts. No per-foot event or per-term per-terrain distribution is stored.')
(out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
np.savez_compressed(out/'series.npz', iterations=np.array(iterations), columns=np.array(columns), values=data,
                    outcomes=counts, families=np.array(families))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors = ['#188557','#3878b5','#e38d21','#b55d99','#cf4548','#7468ac']
labels = ['T1 Touchdown quality','T2 Reference contact match','T3 Slip',
          'T4 Stumble','T5 Contact switching','T6 Contact force']
smooth = 200
x = np.arange(len(rows)).reshape(-1,smooth).mean(1)+1
y = np.column_stack([data[:,index['RewardStep/'+t]].reshape(-1,smooth).mean(1) for t in terms])
fig, axes = plt.subplots(2,1, figsize=(11,7.5), sharex=True, constrained_layout=True)
for j,(label,color) in enumerate(zip(labels,colors)):
    axes[0].plot(x,y[:,j],label=label,color=color,lw=1.6)
axes[0].axhline(0,color='#666666',lw=.7)
axes[0].set_ylabel('Mean reward per env / control step')
axes[0].set_title('Stage2 terrain rewards | signed contributions, weight and dt included')
axes[0].legend(ncol=3,fontsize=9)
axes[0].grid(alpha=.2)
positive = y[:,[0,1]].sum(1)
negative = y[:,2:].sum(1)
axes[1].plot(x,positive,label='T1 + T2',color='#188557')
axes[1].plot(x,negative,label='T3 + T4 + T5 + T6',color='#cf4548')
axes[1].plot(x,y.sum(1),label='Terrain total',color='#252a34',lw=2)
axes[1].axhline(0,color='#666666',lw=.7)
axes[1].set_xlabel('Completed PPO updates (non-overlapping 200-update means)')
axes[1].set_ylabel('Mean reward per env / control step')
axes[1].legend(ncol=3)
axes[1].grid(alpha=.2)
fig.savefig(out/'terrain_reward_trends.png',dpi=180)
fig.savefig(out/'terrain_reward_trends.svg')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
early=np.array([windows[0]['means']['RewardStep/'+t] for t in terms])
late=np.array([windows[-1]['means']['RewardStep/'+t] for t in terms])
xs=np.arange(len(terms))
axes[0].bar(xs-.18,early,width=.36,label='First 1,000 updates',color='#8ba6c5')
axes[0].bar(xs+.18,late,width=.36,label='Last 1,000 updates',color='#244a73')
axes[0].set_xticks(xs,[f'T{i+1}' for i in xs])
axes[0].axhline(0,color='#555555',lw=.8)
axes[0].set_ylabel('Mean signed reward per control step')
axes[0].set_title('Early vs late terrain reward magnitude')
axes[0].legend(fontsize=9)
for group,color in zip(environment['pgmt_reward_groups'],['#3878b5','#e38d21','#188557','#7468ac']):
    series=data[:,index['RewardGroupStep/'+group]].reshape(-1,smooth).mean(1)
    axes[1].plot(x,series,label=group,color=color)
axes[1].axhline(0,color='#555555',lw=.8)
axes[1].set_title('All four reward groups (raw contributions)')
axes[1].set_xlabel('Completed PPO updates')
axes[1].legend(ncol=2)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(out/'reward_scale_comparison.png',dpi=180)
fig.savefig(out/'reward_scale_comparison.svg')
plt.close(fig)
for name in ('first_window','last_window','whole_run'):
    r=summary[name]
    print(name,json.dumps({k:r[k] for k in r if k!='means'}),flush=True)
    print('reward groups',{g:r['means']['RewardGroupStep/'+g] for g in environment['pgmt_reward_groups']},flush=True)
    print('terrain terms',{t:r['means']['RewardStep/'+t] for t in terms},flush=True)
print('Output:',out,flush=True)
