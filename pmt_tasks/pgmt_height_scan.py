"""Tensor contract for the clean, torso-relative PGMT terrain scan."""
import torch


from motion_tracking_rl.pgmt_perception_contract import SCAN_RESOLUTION, SCAN_SHAPE, SCAN_SIZE


def height_scan_from_hits(hits_w, torso_z, map_valid):
    """Convert native RayCaster xy-ordered hits to [N,21,21] maps.

    Positive height means ABOVE the torso, negative means BELOW it. Absolute
    world z and the ray's launch offset do not define the height zero. Invalid
    cells have a finite zero placeholder AND valid=False, never a ground label.
    Diagnostic height_w retains signed world heights and NaN for invalid cells.
    The input sensor buffers are never modified.
    """
    count = SCAN_SHAPE[0] * SCAN_SHAPE[1]
    if hits_w.ndim != 3 or hits_w.shape[1:] != (count, 3):
        raise ValueError(f"PGMT scan requires [N,{count},3] hits, got {tuple(hits_w.shape)}")
    if torso_z.shape != hits_w.shape[:1] or map_valid.shape != hits_w.shape[:2]:
        raise ValueError("PGMT scan torso height/map mask shape does not match ray hits")
    valid = map_valid.bool() & torch.isfinite(hits_w).all(-1) & torch.isfinite(torso_z[:, None])
    shape = (-1, *SCAN_SHAPE)
    return {
        "height": torch.where(valid, hits_w[..., 2] - torso_z[:, None], 0.0).reshape(shape),
        "valid": valid.reshape(shape),
        "height_w": torch.where(valid, hits_w[..., 2], float("nan")).reshape(shape),
    }
