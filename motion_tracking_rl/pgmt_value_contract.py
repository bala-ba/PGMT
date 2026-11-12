"""Semantic order of PGMT value outputs and already-weighted reward groups."""

STAGE1_VALUE_GROUPS = ("upper", "lower", "auxiliary")
TERRAIN_VALUE_GROUPS = ("upper", "lower", "terrain", "auxiliary")


def validate_value_groups(groups):
    if not isinstance(groups, (list, tuple)) or tuple(groups) not in (STAGE1_VALUE_GROUPS, TERRAIN_VALUE_GROUPS):
        raise ValueError("PGMT value_groups must be [upper, lower, auxiliary] or [upper, lower, terrain, auxiliary]")
    return tuple(groups)
