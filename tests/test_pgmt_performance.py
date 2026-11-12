"""Performance options must preserve episode accounting and reward audit coverage."""
from types import SimpleNamespace
import pytest
import torch

from pmt_tasks.pgmt_performance import (
    DeferredRewardAudit, parse_features, scatter_sampler_outcomes,
    tensor_command_reset, tensor_curriculum_reset, tensor_termination_reset,
    tensor_sampler_update_metrics, tensor_sampler_metrics,
)


def test_motion_outcomes_duplicates_empty_and_no_frame_counts():
    sampler = SimpleNamespace(device="cpu", current_failed_motion_count=torch.zeros(4),
                              current_success_motion_count=torch.zeros(4))
    scatter_sampler_outcomes(sampler, torch.tensor([2, 1, 2, 3, 2]), torch.zeros(5),
                             torch.tensor([True, False, False, True, True]))
    assert sampler.current_failed_motion_count.tolist()==[0, 0, 2, 1]
    assert sampler.current_success_motion_count.tolist()==[0, 1, 1, 0]
    scatter_sampler_outcomes(sampler, torch.empty(0,dtype=torch.long), torch.empty(0), torch.empty(0,dtype=torch.bool))
    assert sampler.current_failed_motion_count.sum()==3


def test_reset_logs_are_terminal_snapshots_not_views_of_new_episode():
    ids=torch.tensor([3, 1])
    command=SimpleNamespace(metrics={"error":torch.tensor([9.,2.,8.,4.])},command_counter=torch.ones(4))
    def resample(rows):
        assert torch.equal(rows,ids)
        assert command.metrics['error'][rows].eq(0).all()
        command.metrics['error'][rows]=100
    command._resample=resample
    log=tensor_command_reset(command,ids)
    assert log['error']==3
    assert command.command_counter.tolist()==[1,0,1,0]
    assert command.metrics['error'].tolist()==[9,100,8,100]
    count=torch.tensor(5.)
    manager=SimpleNamespace(_curriculum_state={'terrain':{'success':count}},_class_term_cfgs=[])
    log=tensor_curriculum_reset(manager,ids)
    count.zero_()
    assert log['Curriculum/terrain/success']==5
    seen=[]
    term=SimpleNamespace(_last_episode_dones=torch.tensor([[True,False],[False,True],[True,True]]),
        _term_names=['failure','timeout'],_class_term_cfgs=[SimpleNamespace(func=SimpleNamespace(reset=lambda **kw:seen.append(kw)))])
    log=tensor_termination_reset(term,ids)
    term._last_episode_dones.zero_()
    assert log['Episode_Termination/failure']==pytest.approx(2/3)
    assert torch.equal(seen[0]['env_ids'],ids)


def test_sampler_log_cache_refreshes_after_restore_and_ema():
    sampler=SimpleNamespace(failed_motion_count=torch.tensor([1.,3.]),success_motion_count=torch.tensor([3.,1.]),
                            beta=2.,uniform_ratio=.2,num_motions=2)
    tensor_sampler_update_metrics(sampler)
    first=tensor_sampler_metrics(sampler)
    assert first['total_failed']==4 and first['total_success']==4
    assert first['adaptive_top1_prob']==pytest.approx(.82)
    sampler.failed_motion_count.copy_(torch.tensor([8.,8.]))
    tensor_sampler_update_metrics(sampler)
    assert tensor_sampler_metrics(sampler)['total_failed']==16
    assert first['total_failed']==4


def audit_fixture():
    names=['upper_a','lower_a','terrain_a','auxiliary_a']
    manager=SimpleNamespace(active_terms=names,_step_reward=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]]))
    audit=DeferredRewardAudit(manager,['upper','lower','terrain','auxiliary'],.02,2,'cpu')
    grouped=manager._step_reward*.02
    extras={'reward_groups':grouped,'time_outs':torch.tensor([False,True])}
    result=(None,grouped.sum(-1),torch.tensor([0,1]),extras)
    return audit,result


def test_deferred_audit_retains_all_samples_and_detects_compensating_group_errors():
    audit,result=audit_fixture()
    audit.add(result);audit.add(result)
    assert audit.read()==dict(steps=2,done_events=2,timeout_events=2,terrain_nonzero_samples=4,max_group_sum_error=0.)
    result[3]['reward_groups'][0,0]+=.1
    result[3]['reward_groups'][0,1]-=.1
    audit.add(result)
    with pytest.raises(AssertionError,match='1 control steps'):
        audit.read()


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_deferred_audit_rejects_nonfinite_samples(bad):
    audit,result=audit_fixture()
    result[3]['reward_groups'][1,2]=bad
    audit.add(result)
    with pytest.raises(AssertionError):audit.read()


def test_feature_switches_reject_typos():
    assert not parse_features('none')
    assert parse_features('reset_stats,audit')=={'reset_stats','audit'}
    with pytest.raises(ValueError):parse_features('reset_stat')


def test_fast_contact_reset_preserves_terminal_snapshot_and_other_rows():
    from copy import deepcopy
    from pmt_tasks.pgmt_contact_state import ContactEventState
    original=ContactEventState(4,device='cpu',dt=.02)
    f=torch.zeros(4,2,3)
    original.update(1,f,f)
    original.update(2,f,f)
    f[...,2]=100
    original.update(3,f,f)
    fast=deepcopy(original)
    fast._fast_reset_buffers=True
    frozen=fast.snapshot()
    ids=torch.tensor([3,1])
    original.reset(ids);fast.reset(ids)
    for k,v in original.snapshot().items():
        torch.testing.assert_close(v,fast.snapshot()[k],rtol=0,atol=0,equal_nan=True)
    assert frozen['touchdown'].all()
    assert not fast.snapshot()['touchdown'][ids].any()
    original.update(4,f,f);fast.update(4,f,f)
    for k,v in original.snapshot().items():
        torch.testing.assert_close(v,fast.snapshot()[k],rtol=0,atol=0,equal_nan=True)


def test_fast_reward_and_history_reset_preserve_saved_episode_logs():
    from pmt_tasks.pgmt_fast_placement import reset_reward_sums,reset_history
    manager=SimpleNamespace(_episode_sums={'x':torch.tensor([10.,20.,30.,40.])},
        _env=SimpleNamespace(max_episode_length_s=10),_class_term_cfgs=[])
    ids=torch.tensor([3,1])
    log=reset_reward_sums(manager,ids)
    assert log['Episode_Reward/x']==3
    assert manager._episode_sums['x'].tolist()==[10,0,30,0]
    manager._episode_sums['x'].fill_(99)
    assert log['Episode_Reward/x']==3
    history=SimpleNamespace(_num_pushes=torch.ones(4,dtype=torch.long),_buffer=torch.ones(10,4,5))
    reset_history(history,ids)
    assert history._buffer[:,ids].eq(0).all() and history._buffer[:,[0,2]].eq(1).all()
    assert history._num_pushes.tolist()==[1,0,1,0]
