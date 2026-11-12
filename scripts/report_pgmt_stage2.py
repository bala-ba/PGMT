"""Assemble exact-checkpoint, geometry, training and matched-cost P2 evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pgmt_stage2_checks import equal_state, center_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    scale_file = root/'scale_limits.json'
    scale_limits = json.loads(scale_file.read_text()) if scale_file.is_file() else None
    sizes = (160,) if scale_limits else (160,1024)
    names = ['gui_160','resume_160','restore_160'] + [f'cost_{n}_{mode}' for n in sizes for mode in ('blind','scan_only','perception')]
    states = {name:json.loads((root/name/'training_status.json').read_text()) for name in names}
    assert all(state['status'] == 'completed' for state in states.values())
    for name,expected in [('gui_160',25),('resume_160',30),('restore_160',30)]:
        assert states[name]['next_iteration'] == expected
    resumed = torch.load(root/'resume_160/final.pt',map_location='cpu',weights_only=False)
    restored = torch.load(root/'restore_160/final.pt',map_location='cpu',weights_only=False)
    exact_keys = ['model_state_dict','optimizer_state_dict','environment_state_dict','policy_metadata',
                  'training_contract','iter','next_iteration','learning_rate','tot_timesteps','tot_time','initialization']
    for key in exact_keys:
        equal_state(resumed[key],restored[key])
    np.testing.assert_array_equal(np.load(root/'resume_160/motion_sampling_probabilities.npy'),
                                  np.load(root/'restore_160/motion_sampling_probabilities.npy'))
    initial = torch.load(root/'gui_160/initial.pt',map_location='cpu',weights_only=False)
    trained = torch.load(root/'gui_160/final.pt',map_location='cpu',weights_only=False)
    parameter_changes = {}
    for group,prefix in [('selector','terrain_glimpse.selector'),('patch_encoder','terrain_glimpse.patch_encoder'),('ifm','intent_attention')]:
        parameter_changes[group] = max(float((trained['model_state_dict'][k]-v).abs().max())
            for k,v in initial['model_state_dict'].items() if k.startswith(prefix))
        assert parameter_changes[group] > 0
    progress = [json.loads(line) for line in (root/'gui_160/progress.jsonl').read_text().splitlines()]
    assert len(progress) == 25 and all(v['last_gradient_l1'][group] > 0 for v in progress for group in parameter_changes)
    before, after = np.load(root/'gui_160/debug_before.npz'), np.load(root/'gui_160/debug_after.npz')
    fixed_center_change = float(np.abs(after['center_xy_m']-before['center_xy_m']).max())
    costs = []
    for n in sizes:
        for mode in ('blind','scan_only','perception'):
            name=f'cost_{n}_{mode}'
            state=states[name]
            records=[json.loads(line) for line in (root/name/'progress.jsonl').read_text().splitlines()]
            assert len(records)==4
            collect, update = state['mean_collection_s'],state['mean_update_s']
            costs.append(dict(num_envs=n,profile=mode,collection_s=collect,ppo_update_s=update,
                environment_steps_per_s=n*24/(collect+update),
                peak_torch_allocated_mib=max(r['peak_allocated_mib'] for r in records),
                peak_torch_reserved_mib=max(r['peak_reserved_mib'] for r in records),
                first_approx_kl=records[0]['losses']['approx_kl'],
                gpu_before=state['gpu_before'],gpu_after=state['gpu_after']))
    family_counts = torch.bincount(initial['environment_state_dict']['curriculum']['families'],minlength=5).tolist()
    report = dict(status='passed_160_scope' if scale_limits else 'passed',scale_limits=scale_limits,source_sha256=states['gui_160']['checkpoint_sha256'],
        migration=states['gui_160']['migration_disabled_max_error'],copied_tensors=49,new_tensors=10,
        family_counts=family_counts,physics_steps=states['gui_160']['checks'],
        geometry=json.loads((root/'gui_160/geometry_report.json').read_text()),
        resume=dict(iterations=[25,30,30],exact_payload_keys=exact_keys,probabilities_exact=True,
                    first=states['resume_160']['strict_restore'],second=states['restore_160']['strict_restore'],
                    physical_reset='Restores RNG before fresh physical reset; no PhysX trajectory replay guarantee'),
        updates=25+5+len(costs)*4,total_rollout_transitions=160*24*30+sum(sizes)*24*4*3,
        parameter_changes=parameter_changes,fixed_flat_probe_center_max_change_m=fixed_center_change,
        center_metrics_final_fixed_probe=center_metrics(torch.from_numpy(after['center_xy_m'])),
        valid_patch_fraction_range=[min(v['valid_patch_fraction'] for v in progress),max(v['valid_patch_fraction'] for v in progress)],
        initial_approx_kl=progress[0]['losses']['approx_kl'],
        subsequent_approx_kl_range=[min(v['losses']['approx_kl'] for v in progress[1:]),max(v['losses']['approx_kl'] for v in progress[1:])],
        final_learning_rate=trained['learning_rate'],costs=costs,
        curricula=states['gui_160']['curriculum_outcomes'],device='cuda:1 / RTX 5090 32 GiB',
        caveats=['Finite integration acceptance, not terrain-adaptation convergence.',
                 'Cost means exclude the first update; four updates per profile, same 24-step rollout and PPO settings.',
                 'Torch peaks exclude most PhysX/RTX allocations; total device use is recorded separately.',
                 'GUI uses extra assertions/overlays and is not a throughput baseline.'])
    files = ['scripts/train_pgmt_stage2.py','scripts/train_pgmt_stage2.sh','scripts/validate_pgmt_stage2.sh',
        'scripts/pgmt_stage2_checks.py','scripts/pgmt_process_guard.py','scripts/report_pgmt_stage2.py','motion_tracking_rl/runners/pgmt_runner.py',
        'motion_tracking_rl/utils/pgmt_checkpoint.py','pmt_tasks/agent_cfgs/pgmt.py','pmt_tasks/builder.py',
        'pmt_tasks/pgmt_perception_viewer.py','pmt_tasks/env_cfgs/pgmt/stage2.py','pmt_tasks/pgmt_curriculum.py',
        'pmt_tasks/isaaclab_rl/rsl_rl/vecenv_wrapper.py','configs/task/pgmt_stage2.yaml',
        'configs/obs/pgmt_stage2.yaml','configs/network/pgmt_perception.yaml','configs/sensor/pgmt_height_scan.yaml']
    report['final_implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    report['asset_identity']=trained['environment_state_dict']['curriculum']['identity']
    for xml in (ROOT/'logs/pgmt_p245_final_regression.xml',):
        if xml.is_file():
            suite=ET.parse(xml).getroot().find('testsuite')
            report['regression']=dict(suite.attrib)
    smoke = root/'guard_smoke_160/training_status.json'
    if smoke.is_file():
        smoke_state = json.loads(smoke.read_text())
        assert smoke_state['status'] == 'completed' and smoke_state['next_iteration'] == 31
        report['final_guard_smoke'] = dict(next_iteration=31,completed_updates=1,
                                         strict_restore=smoke_state['strict_restore'])
    report['report_runtime'] = dict(torch=torch.__version__, numpy=np.__version__)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    render(root)
    print(json.dumps({k:report[k] for k in ('status','family_counts','parameter_changes','resume','initial_approx_kl','subsequent_approx_kl_range')},indent=2))


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cmap=deepcopy(plt.get_cmap('viridis'))
    cmap.set_bad('#dbdbdb')
    families=['flat','stairs','slope','boxes','rough']
    fig,axes=plt.subplots(3,5,figsize=(16,10),layout='constrained')
    for row,level in enumerate((0,5,9)):
        data=np.load(root/f'gui_160/geometry_L{level}.npz')
        for column,family in enumerate(families):
            ax=axes[row,column]
            masked=np.ma.masked_where(data['terrain_valid'][column]<.5,data['terrain_map'][column])
            im=ax.imshow(masked,origin='lower',extent=(-1.05,1.05,-1.05,1.05),vmin=-3,vmax=3,cmap=cmap)
            for i,(x,y) in enumerate(data['center_xy_m'][column]):
                ax.add_patch(Rectangle((x-.2,y-.2),.4,.4,fill=False,edgecolor='magenta',linewidth=1.4))
                ax.text(x,y,str(i),color='black',fontsize=9)
            ax.set_title(f'{family} L{level} | valid {data["terrain_valid"][column].mean():.0%}')
            ax.set_xlabel('torso-yaw X (m)')
            if column==0: ax.set_ylabel('torso-yaw Y (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Saved augmented sensor - ground height (m)',shrink=.75)
    fig.suptitle('Actual Isaac observations: 2 x 2 m / 21 x 21; gray = invalid; magenta = 4 glimpse windows')
    fig.savefig(root/'geometry_glimpses.png',dpi=150)
    plt.close(fig)
    data=np.load(root/'gui_160/geometry_L5.npz')
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for i in range(4):
        ax=axes[0,i]
        height=np.ma.masked_where(~data['point_valid'][1,i],data['patch_height'][1,i])
        im=ax.imshow(height,origin='lower',vmin=-1,vmax=1,cmap=cmap)
        ax.set_title(f'Stairs L5 patch {i}, coverage {data["coverage"][1,i]:.2f}')
        axes[1,i].imshow(data['point_valid'][1,i],origin='lower',vmin=0,vmax=1,cmap='gray')
        axes[1,i].set_title('Validity mask')
    fig.colorbar(im,ax=axes[0].tolist(),label='Normalized height (metric / 3)')
    fig.savefig(root/'patch_heights_masks.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,3.8),layout='constrained')
    for j,family in enumerate(families):
        im=axes[j].imshow(data['attention'][j,:,0],vmin=0,vmax=1,cmap='magma',aspect='auto')
        axes[j].axvline(8.5,color='cyan',linewidth=1)
        axes[j].set_title(f'{family} L5')
        axes[j].set_xticks([0,4,8,9,12],['cmd0','cmd4','hist','map0','map3'],rotation=45)
        axes[j].set_yticks(range(4),[f'head{i}' for i in range(4)])
    fig.colorbar(im,ax=axes.tolist(),label='Attention weight')
    fig.savefig(root/'attention_heads.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':
    main()
