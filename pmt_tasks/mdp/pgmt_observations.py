"""ObservationManager adapters for PGMT tracking and terrain perception."""
from pmt_tasks.pgmt_runtime import get_pgmt_state


def pgmt_proprio(env):
    return get_pgmt_state(env).observations()["proprio"]


def pgmt_command_window(env):
    return get_pgmt_state(env).observations()["command_window"]


def pgmt_critic(env):
    return get_pgmt_state(env).observations()["critic"]


def pgmt_terrain_map(env):
    # Legacy placement/P1 geometry interface. P2 uses the atomic term below.
    return get_pgmt_state(env).terrain_scan()["height"]


def pgmt_terrain_valid(env):
    return get_pgmt_state(env).terrain_scan()["valid"].float()


def pgmt_terrain_observation(env):
    from pmt_tasks.pgmt_perception import pack_terrain_observation
    return pack_terrain_observation(get_pgmt_state(env).terrain_observations())
