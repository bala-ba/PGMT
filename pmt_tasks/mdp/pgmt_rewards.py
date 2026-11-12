"""RewardManager adapter. Each configured term is an unweighted reward rate."""
from pmt_tasks.pgmt_runtime import get_pgmt_state


def pgmt_reward(env, name: str):
    return get_pgmt_state(env).rewards()[name]
