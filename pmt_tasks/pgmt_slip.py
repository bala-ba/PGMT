"""T3: sum of squared world-XY ankle-link speeds during actual contact.

Returns a nonnegative, unweighted cost rate. RewardManager alone applies -1
and the control dt. Reference labels, touchdown events and terrain maps are
not inputs to this continuous penalty.
"""
import torch


SLIP_WEIGHT = -1.0


def empty_slip(num_envs, *, device):
    return dict(
        contact=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        contact_valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        velocity_w=torch.full((num_envs, 2, 3), float("nan"), device=device),
        velocity_valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        active=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        speed_xy_sq=torch.full((num_envs, 2), float("nan"), device=device),
        per_foot=torch.zeros(num_envs, 2, device=device),
        rate=torch.zeros(num_envs, device=device),
    )


@torch.no_grad()
def evaluate_slip(contact, contact_valid, velocity_w):
    """[N,2] states and [N,2,3] link-origin velocity -> [N] cost rate.

    Invalid data is explicitly unknown, not a valid stationary foot. Replace
    nonfinite rows before arithmetic to avoid NaN * 0; finite velocities have
    no dead zone, clipping, averaging over feet, or normal/tangent projection.
    """
    if (contact.ndim != 2 or contact.shape[1] != 2 or contact_valid.shape != contact.shape
            or velocity_w.shape != (*contact.shape, 3)
            or contact.dtype != torch.bool or contact_valid.dtype != torch.bool
            or not velocity_w.is_floating_point()
            or contact.device != velocity_w.device or contact_valid.device != contact.device):
        raise ValueError("Slip requires boolean [N,2] contact/mask and floating [N,2,3] velocity on one device")
    velocity_valid = torch.isfinite(velocity_w).all(-1)
    safe = torch.where(velocity_valid[..., None], velocity_w[..., :2], 0.)
    speed_sq = safe.square().sum(-1)
    velocity_valid &= torch.isfinite(speed_sq)
    valid = contact_valid & velocity_valid
    active = valid & contact
    per_foot = torch.where(active, speed_sq, 0.)
    # Overflow is diagnosed as invalid, never silently clipped into a plausible speed.
    sum_valid = torch.isfinite(per_foot.sum(-1))
    valid &= sum_valid[:, None]
    active &= sum_valid[:, None]
    per_foot = torch.where(active, speed_sq, 0.)
    return dict(contact=contact.clone(), contact_valid=contact_valid.clone(), velocity_w=velocity_w.clone(),
                velocity_valid=velocity_valid, valid=valid, active=active,
                speed_xy_sq=torch.where(velocity_valid, speed_sq, float("nan")),
                per_foot=per_foot, rate=per_foot.sum(-1))
