"""T2: continuous, validity-masked match to the same-frame reference contacts.

Actual contact is the shared all-object normal-force norm > 10 N state.
Only RewardManager applies the +1.5 weight and the control dt.
"""
import torch


REFERENCE_CONTACT_MATCH_WEIGHT = 1.5


def empty_contact_match(num_envs, *, device):
    result = {key: torch.zeros(num_envs, 2, dtype=torch.bool, device=device) for key in (
        "actual_contact", "actual_valid", "reference_contact", "reference_valid", "compared_valid", "matched")}
    result["reference_count"] = torch.zeros(num_envs, dtype=torch.long, device=device)
    result["rate"] = torch.zeros(num_envs, device=device)
    return result


@torch.no_grad()
def evaluate_reference_contact_match(contact, contact_valid, reference_contact, reference_valid):
    """Return [N,2] diagnostics and an unweighted [N] rate in [0,1].

    The denominator counts valid reference labels, as in the recorded formula.
    An unknown physical measurement cannot earn a match or shrink this
    denominator; it remains explicitly unknown in compared_valid. Invalid
    reference labels are excluded entirely, never interpreted as swing.
    """
    inputs = (contact, contact_valid, reference_contact, reference_valid)
    if (contact.ndim != 2 or contact.shape[1] != 2
            or any(value.shape != contact.shape or value.device != contact.device for value in inputs)
            or any(value.dtype != torch.bool for value in (contact, contact_valid, reference_valid))):
        raise ValueError("Contact match requires paired [N,2] states/labels and boolean masks on one device")
    # The label store validates binary metadata at load time. Keep the kernel
    # finite even if a diagnostic caller supplies NaN/Inf/unknown state codes.
    ref_valid = reference_valid & ((reference_contact == 0) | (reference_contact == 1))
    ref = reference_contact == 1
    valid = contact_valid & ref_valid
    matched = valid & (contact == ref)
    count = ref_valid.sum(-1)
    rate = matched.sum(-1).float() / count.clamp_min(1)
    return dict(actual_contact=contact.clone(), actual_valid=contact_valid.clone(),
                reference_contact=ref, reference_valid=ref_valid, compared_valid=valid,
                matched=matched, reference_count=count, rate=rate)
