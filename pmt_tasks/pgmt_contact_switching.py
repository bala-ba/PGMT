"""T5: count shared fast-contact-switch events without advancing any timer.

The shared contact state owns both edge detection and age/grace thresholds.
RewardManager alone applies the negative weight and control dt.
"""
import torch


CONTACT_SWITCHING_WEIGHT = -30.0


def empty_contact_switching(num_envs, *, device):
    return dict(
        fast_contact_switch=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        per_foot=torch.zeros(num_envs, 2, device=device),
        rate=torch.zeros(num_envs, device=device),
    )


@torch.no_grad()
def evaluate_contact_switching(fast_contact_switch, contact_history_valid):
    """Boolean [N,2] event and same-episode comparison validity -> [N] cost.

    Unknown measurements/baselines cannot be switch events. No reference label,
    terrain-support gate, new age counter, or second float-time comparison.
    """
    if (fast_contact_switch.ndim != 2 or fast_contact_switch.shape[1] != 2
            or contact_history_valid.shape != fast_contact_switch.shape
            or fast_contact_switch.dtype != torch.bool or contact_history_valid.dtype != torch.bool
            or fast_contact_switch.device != contact_history_valid.device):
        raise ValueError("Contact switching requires boolean [N,2] events and history validity on one device")
    valid = contact_history_valid.clone()
    per_foot = (fast_contact_switch & valid).float()
    return dict(fast_contact_switch=fast_contact_switch.clone(), valid=valid,
                per_foot=per_foot, rate=per_foot.sum(-1))
