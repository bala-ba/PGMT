"""Isaac-independent PGMT version-3 observation contract shared by task/policy."""
import math

SCAN_SIZE = (2.0, 2.0)
SCAN_RESOLUTION = 0.1
SCAN_SHAPE = (21, 21)  # row=y, col=x; each axis increases from -1 m to +1 m


def terrain_observation_contract(*, augmentation_enabled=True, nan_dropout_prob=0.1,
                                 noise_range_m=(-0.06, 0.06)):
    """Version 3: PMT sign/fill, metric observations and fresh per-call noise.

    Disabling augmentation disables only dropout/noise, not sign, fill or clip.
    Normalization belongs to the network, not this observation function.
    """
    if not isinstance(augmentation_enabled, bool):
        raise ValueError("PGMT augmentation_enabled must be bool")
    nan_dropout_prob = float(nan_dropout_prob)
    if not math.isfinite(nan_dropout_prob) or not 0 <= nan_dropout_prob <= 1:
        raise ValueError("PGMT nan_dropout_prob must be finite and in [0, 1]")
    if not isinstance(noise_range_m, (list, tuple)) or len(noise_range_m) != 2:
        raise ValueError("PGMT noise_range_m must contain two finite ordered values")
    noise_range_m = [float(value) for value in noise_range_m]
    if not all(math.isfinite(value) for value in noise_range_m) or noise_range_m[0] > noise_range_m[1]:
        raise ValueError("PGMT noise_range_m must contain two finite ordered values")
    return dict(version=3, packed_key="pgmt_terrain", map_key="terrain_map", valid_key="terrain_valid",
                shape=list(SCAN_SHAPE), size_m=list(SCAN_SIZE), resolution_m=SCAN_RESOLUTION,
                frame="actual_torso_yaw", ordering="row=y,col=x; -1..+1 m",
                height="sensor_world_z - ground_world_z - offset_m", offset_m=0.0,
                units="m", dtype="float32", invalid_world_z_fill_m=30.0,
                invalid_sensor_z_fill_m=0.78, mask="float32 0/1; map_valid AND finite_hit_z AND NOT dropped",
                augmentation_enabled=augmentation_enabled, nan_dropout_prob=nan_dropout_prob,
                noise="independent_uniform", noise_range_m=noise_range_m, clipping_m=[-3.0, 3.0],
                normalization_location="network", network_z_clip_m=3.0, network_normalize_height=True,
                noise_refresh="every_observation_function_call")


def validate_terrain_observation_contract(config):
    required = ("augmentation_enabled", "nan_dropout_prob", "noise_range_m")
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ValueError("Missing or obsolete PGMT terrain observation schema (requires version 3)")
    if config != terrain_observation_contract(**{key: config[key] for key in required}):
        raise ValueError("Unsupported PGMT terrain observation schema/coordinates/noise contract")
