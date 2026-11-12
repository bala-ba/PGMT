"""Real two-process PPO, checkpoint/resume and coordinated-stop regressions."""
from datetime import timedelta
import faulthandler
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from scripts.pgmt_distributed import RankLayout, prepare_launch
from motion_tracking_rl.networks.layers.normalization import EmpiricalNormalization
from motion_tracking_rl.runners import PGMTStage2Runner
from motion_tracking_rl.utils.pgmt_distributed import coordinated_stop, cpu_state, select_rank_state


def test_launch_preserves_single_device_and_maps_explicit_gpu_list():
    command = ['python', 'train.py', '--device', 'cuda:1', '--num_envs', '8192']
    assert prepare_launch(command, {}) == (command, {})
    launched, env = prepare_launch([*command, '--gpus', '1,0'], {})
    assert launched[:3] == ['python', '-m', 'torch.distributed.run']
    assert '--nproc_per_node=2' in launched and '--max_restarts=0' in launched
    assert launched[-2:] == ['--device', 'cuda:0']
    assert '--gpus' not in launched and env['CUDA_VISIBLE_DEVICES'] == '1,0'
    launched, env = prepare_launch([*command, '--gpus=1'], {})
    assert launched[:2] == command[:2] and env['CUDA_VISIBLE_DEVICES'] == '1'
    assert launched[-2:] == ['--device', 'cuda:0']


@pytest.mark.parametrize('gpus', ['', '0,0', '0,00', '-1,0', '0,', 'cuda:0,1'])
def test_invalid_gpu_selection(gpus):
    with pytest.raises(SystemExit):
        prepare_launch(['python', 'train.py', f'--gpus={gpus}'], {})


def test_rank_partition_and_output_paths():
    layout = RankLayout.from_environment(dict(WORLD_SIZE='2', RANK='1', LOCAL_RANK='1'))
    assert layout.local_envs(8192) == 4096
    assert layout.output_dir('/tmp/run') == Path('/tmp/run/ranks/rank_1')
    assert RankLayout().local_envs(8192) == 8192
    assert RankLayout().output_dir('/tmp/run') == Path('/tmp/run')
    for total in (8191, 8):
        with pytest.raises(ValueError):
            layout.local_envs(total)


def test_nccl_override_is_only_applied_to_distributed_children(tmp_path):
    library = tmp_path / 'libnccl.so.2'
    library.touch()
    env = dict(PGMT_NCCL_LIBRARY=str(library), LD_PRELOAD='/existing.so')
    command = ['python', 'train.py']
    assert prepare_launch(command, env)[1]['LD_PRELOAD'] == '/existing.so'
    assert prepare_launch([*command, '--gpus', '1'], env)[1]['LD_PRELOAD'] == '/existing.so'
    assert prepare_launch([*command, '--gpus', '0,1'], env)[1]['LD_PRELOAD'] == f'{library}:/existing.so'
    assert env['LD_PRELOAD'] == '/existing.so'
    library.unlink()
    with pytest.raises(ValueError, match='does not exist'):
        prepare_launch([*command, '--gpus', '0,1'], env)


def test_checkpoint_topology_validation():
    with pytest.raises(ValueError, match='single-GPU'):
        select_rank_state({}, 2, 0)
    with pytest.raises(ValueError, match='world size'):
        select_rank_state(dict(distributed=dict(version=1, world_size=2, rank_states=[])), 1, 0)
    with pytest.raises(ValueError, match='ordering'):
        select_rank_state(dict(distributed=dict(version=1, world_size=2,
                          rank_states=[dict(rank=1), dict(rank=0)])), 2, 0)


class TinyEnv:
    num_envs, num_actions = 6, 29
    max_episode_length = 20

    def __init__(self, device, rank):
        self.device, self.rank = device, rank
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.state = dict(rank=rank, steps=0)
        self.obs = self._observations()

    def _observations(self):
        n, device = self.num_envs, self.device
        return TensorDict(dict(proprio=torch.randn(n, 96, device=device) + self.rank,
            critic=torch.randn(n, 251, device=device), proprio_history=torch.randn(n, 10, 96, device=device),
            command_window=torch.randn(n, 8, 61, device=device), terrain_map=torch.randn(n, 21, 21, device=device),
            terrain_valid=torch.ones(n, 21, 21, device=device)), [n])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.state['steps'] += 1
        self.obs = self._observations()
        groups = torch.randn(self.num_envs, 4, device=self.device)
        return self.obs, groups.sum(-1), torch.zeros(self.num_envs, device=self.device), {'reward_groups': groups}

    def get_checkpoint_state(self):
        return dict(self.state)

    def validate_checkpoint_state(self, state):
        assert state['rank'] == self.rank and state['steps'] >= 0

    def load_checkpoint_state(self, state):
        self.validate_checkpoint_state(state)
        self.state = dict(state)


class GlooRunner(PGMTStage2Runner):
    """Use the production runner/algorithm with a CPU transport for CI."""
    def _configure_multi_gpu(self):
        self.gpu_global_rank = self.gpu_local_rank = dist.get_rank()
        self.gpu_world_size = dist.get_world_size()
        self.is_distributed = True
        self.multi_gpu_cfg = dict(global_rank=self.gpu_global_rank,
                                  local_rank=self.gpu_local_rank, world_size=self.gpu_world_size)


def make_runner(device, rank, directory):
    groups = {name: [key] for name, key in [('policy', 'proprio'), ('critic', 'critic'),
        ('policy_history', 'proprio_history'), ('command_window', 'command_window'),
        ('terrain_map', 'terrain_map'), ('terrain_valid', 'terrain_valid')]}
    cfg = dict(num_steps_per_env=2, save_interval=1, logger='tensorboard', obs_groups=groups,
        pgmt_training_contract=dict(rank=rank, num_envs=6, world_size=2),
        algorithm=dict(class_name='PPO', learning_rate=1e-4, schedule='adaptive', desired_kl=.01,
                       num_learning_epochs=2, num_mini_batches=2),
        policy=dict(class_name='PGMTActorCritic', actor_hidden_dims=[16], critic_hidden_dims=[16],
                    actor_obs_normalization=True, critic_obs_normalization=True,
                    terrain_glimpse_cfg={}, terrain_mode='enabled',
                    value_groups=['upper', 'lower', 'terrain', 'auxiliary']))
    cls = GlooRunner if device == 'cpu' else PGMTStage2Runner
    result = cls(TinyEnv(device, rank), cfg, log_dir=str(directory), device=device)
    result.git_status_repos = []
    return result


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def distributed_worker(rank, directory, use_cuda):
    faulthandler.dump_traceback_later(35)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    device = f'cuda:{rank}' if use_cuda else 'cpu'
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if use_cuda else 'gloo',
                            init_method=f'file://{directory}/rendezvous', rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    directory = Path(directory)
    model = fresh = None
    try:
        # Unequal local batch sizes exercise weighted moment pooling.
        norm, reference = EmpiricalNormalization(3).to(device), EmpiricalNormalization(3).to(device)
        norm.sync_distributed = True
        for offset in (0., 100.):
            batches = [torch.arange(6, device=device).reshape(2, 3).float() + offset,
                       torch.arange(9, device=device).reshape(3, 3).float() + offset + 10.]
            with torch.inference_mode():
                norm.update(batches[rank])
                reference.update(torch.cat(batches))
            for key in norm.state_dict():
                torch.testing.assert_close(norm.state_dict()[key], reference.state_dict()[key])

        torch.manual_seed(722 + rank)
        model = make_runner(device, rank, directory)
        model.initialization = {'source': 'synthetic regression'}
        model.alg.broadcast_parameters()
        model.save(directory / 'initial.pt')  # Collective save before any updates.
        model.learn(2)
        assert model.completed_iterations == 2
        states = [None, None]
        dist.all_gather_object(states, cpu_state(dict(model=model.alg.policy.state_dict(),
                                                     optimizer=model.alg.optimizer.state_dict())))
        assert_same(states[0], states[1])
        model.save(directory / 'final.pt')
        dist.barrier()
        saved = torch.load(directory / 'final.pt', map_location='cpu', weights_only=False)
        assert saved['distributed']['world_size'] == 2
        assert saved['next_iteration'] == 2
        assert saved['tot_timesteps'] == 2 * 2 * 6 * 2
        assert [s['environment_state_dict']['rank'] for s in saved['distributed']['rank_states']] == [0, 1]
        assert not torch.equal(saved['distributed']['rank_states'][0]['rng_state']['torch'],
                               saved['distributed']['rank_states'][1]['rng_state']['torch'])
        fresh = make_runner(device, rank, directory)
        fresh.load(directory / 'final.pt')
        assert_same(fresh.alg.policy.state_dict(), model.alg.policy.state_dict())
        assert_same(fresh.alg.optimizer.state_dict(), model.alg.optimizer.state_dict())
        assert fresh.env.state == model.env.state
        assert fresh.completed_iterations == 2

        class Stop(Exception):
            pass
        def callback(locs):
            reason = coordinated_stop('rank_one_stop' if rank else None, device, True)
            assert reason is not None
            raise Stop(reason)
        fresh.iteration_callback = callback
        with pytest.raises(Stop):
            fresh.learn(5)
        assert fresh.completed_iterations == 3
        fresh.save(directory / 'stopped.pt')
        dist.barrier()
        assert torch.load(directory / 'stopped.pt', map_location='cpu', weights_only=False)['next_iteration'] == 3
    finally:
        for runner in (model, fresh):
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize('use_cuda', [False, pytest.param(True, marks=pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='Two CUDA devices required'))], ids=['gloo', 'nccl'])
def test_two_rank_training_resume_and_stop(tmp_path, use_cuda):
    context = mp.spawn(distributed_worker, args=(str(tmp_path), use_cuda), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Distributed training exceeded 120 seconds (possible collective deadlock)')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
