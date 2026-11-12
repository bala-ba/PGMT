"""Device-resident reference contact labels, independent of Isaac and reset eligibility."""
import numpy as np
import torch

CONTACT_FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


class ContactLabelStore:
    """Pack unchunked motion metadata in exactly the command loader's order.

    Only contacts and contact_valid are authoritative; stance/swing and state
    codes are derived after masking and canonical left/right reordering. Labels
    describe the source reference, not measured forces on the training terrain.
    """

    foot_names = CONTACT_FOOT_NAMES

    def __init__(self, motions, *, device):
        if not motions:
            raise ValueError("ContactLabelStore requires at least one motion")
        contacts, validity, lengths = [], [], []
        self.source_files = tuple(motion.source_file for motion in motions)
        for motion in motions:
            metadata = motion.metadata
            missing = {"contacts", "contact_valid", "foot_names"} - metadata.keys()
            if missing:
                raise ValueError(f"Missing contact metadata {sorted(missing)}: {motion.source_file}")
            names = np.asarray(metadata["foot_names"])
            if names.shape != (2,) or set(names.tolist()) != set(self.foot_names):
                raise ValueError(f"Expected named left/right ankle contact labels: {motion.source_file}")
            order = [names.tolist().index(name) for name in self.foot_names]
            contact = np.asarray(metadata["contacts"])
            valid = np.asarray(metadata["contact_valid"])
            expected = (motion.num_frames, 2)
            if motion.num_frames < 1 or contact.shape != expected or valid.shape != expected:
                raise ValueError(f"Contact arrays must match unchunked motion shape {expected}: {motion.source_file}")
            if not np.isin(contact, [0, 1]).all() or not np.isin(valid, [0, 1]).all():
                raise ValueError(f"Contact labels and validity must be finite binary values: {motion.source_file}")
            contacts.append(contact[:, order].astype(bool))
            validity.append(valid[:, order].astype(bool))
            lengths.append(motion.num_frames)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = self.motion_lengths.cumsum(0) - self.motion_lengths
        self._contacts = torch.as_tensor(np.concatenate(contacts), dtype=torch.bool, device=device)
        self._valid = torch.as_tensor(np.concatenate(validity), dtype=torch.bool, device=device)

    def gather(self, motion_ids, frame_ids):
        """Return fresh tensors for any matching index shape, e.g. [N] or [N,8].

        Out-of-range frame requests read the nearest boundary label but have
        frame_valid=False and contact_valid=False. Thus future pose clamping
        never fabricates valid contact supervision beyond a segment's end.
        Motion ids must be valid loader indices; they are never clamped/wrapped.
        No per-step cache, host transfers or file reads are used in this path.
        """
        if motion_ids.shape != frame_ids.shape or motion_ids.dtype != torch.long or frame_ids.dtype != torch.long:
            raise ValueError("Expected matching int64 motion/frame index tensors")
        if motion_ids.device != self.motion_lengths.device or frame_ids.device != self.motion_lengths.device:
            raise ValueError("Contact indices must be on the label store device")
        shape = motion_ids.shape
        ids = motion_ids.reshape(-1)
        lengths = self.motion_lengths.index_select(0, ids).reshape(shape)
        frame_valid = (frame_ids >= 0) & (frame_ids < lengths)
        frames = torch.minimum(frame_ids.clamp_min(0), lengths - 1)
        linear = self.offsets.index_select(0, ids).reshape(shape) + frames
        contact = self._contacts[linear]
        valid = self._valid[linear] & frame_valid[..., None]
        stance, swing = valid & contact, valid & ~contact
        foot_state = torch.where(valid, contact.to(torch.int8), -1)
        support_mode = contact[..., 0].to(torch.int8) + 2 * contact[..., 1].to(torch.int8)
        support_mode = torch.where(valid.all(-1), support_mode, -1)
        return dict(contacts=contact.float(), contact_valid=valid, stance=stance, swing=swing,
                    foot_state=foot_state, support_mode=support_mode, frame_valid=frame_valid,
                    motion_ids=motion_ids.clone(), frame_ids=frames, requested_frame_ids=frame_ids.clone())
