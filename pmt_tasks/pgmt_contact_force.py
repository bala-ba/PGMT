"""T6: sum each foot's squared all-object normal contact-force magnitude.

No contact threshold, force allowance, event gate or averaging over feet.
RewardManager alone applies -1e-6 and the control dt to this positive cost.
"""
import torch


CONTACT_FORCE_WEIGHT = -1e-6


def empty_contact_force(num_envs, *, device):
    return dict(
        forces_w=torch.full((num_envs, 2, 3), float("nan"), device=device),
        force_valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        force_sq=torch.full((num_envs, 2), float("nan"), device=device),
        valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        per_foot=torch.zeros(num_envs, 2, device=device),
        rate=torch.zeros(num_envs, device=device),
    )


@torch.no_grad()
def evaluate_contact_force(forces_w, force_valid):
    """[N,2,3] world forces and [N,2] measurement validity -> [N] cost.

    force_valid is measurement validity, not the norm > 10 N contact state.
    Invalid rows are masked before squaring. Nonfinite squared magnitudes or
    aggregate overflow are explicitly invalid, never clipped into valid data.
    """
    if (forces_w.ndim != 3 or forces_w.shape[1:] != (2, 3)
            or force_valid.shape != forces_w.shape[:2] or force_valid.dtype != torch.bool
            or not forces_w.is_floating_point() or forces_w.device != force_valid.device):
        raise ValueError("Contact force requires floating [N,2,3] forces and boolean [N,2] validity on one device")
    valid = force_valid & torch.isfinite(forces_w).all(-1)
    safe = torch.where(valid[..., None], forces_w, 0.)
    force_sq = safe.square().sum(-1)
    valid &= torch.isfinite(force_sq)
    diagnostic = torch.where(valid, force_sq, float("nan"))
    per_foot = torch.where(valid, force_sq, 0.)
    valid &= torch.isfinite(per_foot.sum(-1))[:, None]
    per_foot = torch.where(valid, force_sq, 0.)
    return dict(forces_w=forces_w.clone(), force_valid=force_valid.clone(), force_sq=diagnostic,
                valid=valid, per_foot=per_foot, rate=per_foot.sum(-1))
