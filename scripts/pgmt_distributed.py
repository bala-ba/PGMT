"""Simulator-free launch and rank layout for PGMT training."""
import argparse
from dataclasses import dataclass
from pathlib import Path


def distributed_environment(environment):
    environment = dict(environment)
    library = environment.get('PGMT_NCCL_LIBRARY')
    if library:
        if not Path(library).is_file():
            raise ValueError(f'PGMT_NCCL_LIBRARY does not exist: {library}')
        # Override only this launch. In particular do not replace Conda packages
        # used by an already-running single-GPU training process.
        environment['LD_PRELOAD'] = library + (':' + environment['LD_PRELOAD']
                                               if environment.get('LD_PRELOAD') else '')
    return environment


def prepare_launch(command, environment):
    """Consume --gpus before Isaac imports; torchrun owns the worker group."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--gpus')
    options, remaining = parser.parse_known_args(command[2:])
    if options.gpus is None:
        return command, (distributed_environment(environment)
                         if int(environment.get('WORLD_SIZE', '1')) > 1 else environment)
    ids = options.gpus.split(',')
    if not ids or any(not value.isdecimal() for value in ids) or len(set(map(int, ids))) != len(ids):
        parser.error('--gpus must contain distinct GPU indices, for example 0,1')
    if int(environment.get('WORLD_SIZE', '1')) > 1:
        parser.error('--gpus cannot be nested inside an existing torchrun launch')
    environment = dict(environment, CUDA_VISIBLE_DEVICES=','.join(str(int(i)) for i in ids))
    # Visible devices are renumbered from zero. This overrides shell defaults.
    worker = [command[1], *remaining, '--device', 'cuda:0']
    if len(ids) == 1:
        return [command[0], *worker], environment
    return [command[0], '-m', 'torch.distributed.run', '--standalone', '--local_addr=127.0.0.1',
            '--nnodes=1', f'--nproc_per_node={len(ids)}', '--max_restarts=0', *worker], distributed_environment(environment)


@dataclass(frozen=True)
class RankLayout:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @classmethod
    def from_environment(cls, environment):
        result = cls(*(int(environment.get(key, default)) for key, default in
                       [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')]))
        if result.world_size < 1 or not 0 <= result.rank < result.world_size or result.local_rank < 0:
            raise ValueError('Invalid distributed rank configuration')
        return result

    @property
    def distributed(self):
        return self.world_size > 1

    def local_envs(self, total):
        if total % self.world_size or total // self.world_size < 5:
            raise ValueError('--num_envs is the global total: it must divide evenly across GPUs, with >=5 per GPU')
        return total // self.world_size

    def output_dir(self, root):
        root = Path(root)
        return root if self.rank == 0 else root / 'ranks' / f'rank_{self.rank}'
