"""PGMT distributed checkpoint selection and coordinated stopping."""
import torch


def cpu_state(value):
    """Avoid serializing tensors tied to a different rank's CUDA device."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return value


def select_rank_state(saved, world_size, rank):
    """Validate topology before loading any model/optimizer/environment state."""
    distributed = saved.get('distributed')
    if distributed is None:
        if world_size != 1:
            raise ValueError('Strict resume requires the same world size; this is a single-GPU checkpoint')
        return saved
    if (distributed.get('version') != 1 or distributed.get('world_size') != world_size
            or len(distributed.get('rank_states', [])) != world_size):
        raise ValueError('Strict resume requires the same distributed world size and all rank states')
    states = distributed['rank_states']
    if [state.get('rank') for state in states] != list(range(world_size)):
        raise ValueError('Invalid distributed checkpoint rank ordering')
    selected = dict(saved)
    for key in ('training_contract', 'environment_state_dict', 'rng_state'):
        selected[key] = states[rank][key]
    return selected


def coordinated_stop(reason, device, distributed):
    """All workers leave at the same completed-update boundary."""
    if not distributed:
        return reason
    flag = torch.tensor(int(reason is not None), device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return (reason or 'peer_stop_request') if flag.item() else None
