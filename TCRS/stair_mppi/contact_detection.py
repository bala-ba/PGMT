"""TCRS height/speed contact detection, shared by demos and offline labels.

The phase clock is deliberately not used to replace these masks. Defaults and
the two-pass run filter are the original MotionClip implementation.
"""
from dataclasses import dataclass

import numpy as np

CONTACT_Z_THRESHOLD = 0.06
CONTACT_SPEED_THRESHOLD = 0.5


def stabilize_contact_mask(mask, min_contact_frames=4, min_swing_frames=2):
    out = np.asarray(mask, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] and (j - i) < min_contact_frames:
            out[i:j] = False
        i = j
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (not out[i]) and (j - i) < min_swing_frames:
            left_true = (i > 0) and out[i - 1]
            right_true = (j < n) and out[j]
            if left_true and right_true:
                out[i:j] = True
        i = j
    return out


@dataclass
class ContactDetection:
    raw: np.ndarray
    contacts: np.ndarray
    height_threshold: float


def detect_foot_contacts(foot_height, foot_speed, *,
                         contact_z_threshold=CONTACT_Z_THRESHOLD,
                         contact_speed_threshold=CONTACT_SPEED_THRESHOLD,
                         min_contact_frames=4, min_swing_frames=2, valid_mask=None):
    """Classify [frames, left/right] ankle heights and world speed magnitudes.

    A caller with a terrain mesh supplies ankle height relative to that mesh.
    MotionClip supplies world z, preserving its historical flat-ground result.
    """
    height = np.asarray(foot_height)
    speed = np.asarray(foot_speed)
    if height.ndim != 2 or height.shape[1] != 2 or speed.shape != height.shape or len(height) == 0:
        raise ValueError("Expected nonempty foot height/speed arrays of shape [frames, 2]")
    if not np.isfinite(height).all() or not np.isfinite(speed).all() or (speed < 0).any():
        raise ValueError("Foot heights/speeds must be finite, with nonnegative speed")
    if contact_z_threshold <= 0 or contact_speed_threshold <= 0 or min(min_contact_frames, min_swing_frames) < 1:
        raise ValueError("Contact thresholds and minimum run lengths must be positive")
    valid = np.ones(height.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != height.shape:
        raise ValueError("valid_mask must have shape [frames, 2]")
    percentile = float(np.percentile(height[valid], 25.0)) + 0.02 if valid.any() else 0.0
    threshold = max(float(contact_z_threshold), percentile)
    raw = valid & (height < threshold) & (speed < float(contact_speed_threshold))
    contacts = np.column_stack([
        stabilize_contact_mask(raw[:, foot], min_contact_frames, min_swing_frames) for foot in range(2)
    ])
    return ContactDetection(raw, contacts & valid, threshold)
