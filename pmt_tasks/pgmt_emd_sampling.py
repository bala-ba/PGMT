"""P-BFM clip priorities and uniform within-clip starts (no frame curriculum)."""
from copy import deepcopy
import math

import numpy as np
import torch


EMD_DEFAULTS = dict(min_value=0.5, max_value=2.0, scale=2.0,
                    evaluation_interval=9_600_000, evaluation_num_envs=64,
                    evaluation_seed=1729)


def emd_config(overrides=None):
    values = dict(EMD_DEFAULTS)
    if overrides is not None:
        unknown = set(overrides) - values.keys()
        if unknown:
            raise ValueError(f"Unknown EMD sampling settings: {sorted(unknown)}")
        values.update(overrides)
    low, high, scale = (values[k] for k in ("min_value", "max_value", "scale"))
    if not all(math.isfinite(v) for v in (low, high, scale)) or not 0 <= low <= high or scale <= 0 or high * scale > 100:
        raise ValueError("Invalid EMD priority bounds/scale")
    for key in ("evaluation_interval", "evaluation_num_envs", "evaluation_seed"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key.endswith("seed") else 1):
            raise ValueError(f"Invalid EMD {key}")
    return values


def joint_trajectory_emd(actual, reference):
    """Exact uniform-mass OT with Euclidean joint-angle cost, as in P-BFM.

    Both trajectories contain all N frames. The equal-mass N x N transport
    optimum is a permutation, so assignment / N equals ot.emd2 exactly;
    this uses the existing SciPy dependency without requiring POT/Sinkhorn.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (actual.ndim != 2 or actual.shape != reference.shape or not all(actual.shape)
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError("EMD requires finite, complete, equally sized joint trajectories")
    cost = cdist(actual, reference, metric="euclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def sample_emd_starts(weights, compatible, lengths, terrain_ids, *, fixed_start_frame=-1):
    """Select a compatible clip first, then uniformly sample its time grid.

    PMT references use integer control frames: floor(U[0, (N-1)/fps) * fps)
    is uniform over 0..N-2. Excluding the endpoint leaves a real transition.
    """
    if not len(terrain_ids):
        return terrain_ids.clone(), terrain_ids.clone()
    available = lengths > (fixed_start_frame if fixed_start_frame >= 0 else 1)
    groups, inverse = terrain_ids.unique(return_inverse=True)
    eligible = compatible[groups] & available[None]
    probabilities = eligible * weights[None]
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities.sum(-1) <= 0).any():
        raise ValueError("No positive EMD priority in a compatible motion pool")
    motions = torch.empty_like(terrain_ids)
    for group in range(len(groups)):
        ids = (inverse == group).nonzero().flatten()
        motions[ids] = torch.multinomial(probabilities[group], len(ids), replacement=True)
    frames = (torch.full_like(motions, fixed_start_frame) if fixed_start_frame >= 0 else
              (torch.rand(len(motions), device=motions.device) * (lengths[motions] - 1)).long())
    return motions, frames


class EMDMotionSampler:
    """Weights change only after a successful complete-library evaluation."""
    kind = "pbfm_emd"
    enable_frame_sampling = False

    def __init__(self, num_motions, device, config=None):
        self.num_motions, self.device = num_motions, device
        self.config = emd_config(config)
        self.weights = torch.ones(num_motions, device=device)
        self.emd = torch.zeros(num_motions, device=device)
        self.last_evaluation_step = 0
        self.evaluation_count = 0

    def update(self, motion_ids, frame_ids, terminated):
        # Episode failures do not alter EMD priorities.
        pass

    def step(self):
        pass

    def due(self, global_step):
        interval = self.config["evaluation_interval"]
        return global_step // interval > self.last_evaluation_step // interval

    def update_emd(self, values, global_step):
        values = torch.as_tensor(values, dtype=self.emd.dtype, device=self.device)
        if (values.shape != self.emd.shape or not torch.isfinite(values).all() or (values < 0).any()
                or isinstance(global_step, bool) or not isinstance(global_step, int)
                or global_step < self.last_evaluation_step):
            raise ValueError("EMD update requires every motion, finite nonnegative scores and a nondecreasing step")
        weights = torch.exp2(values.clamp(self.config["min_value"], self.config["max_value"]) * self.config["scale"])
        self.emd.copy_(values)
        self.weights.copy_(weights)
        self.last_evaluation_step = global_step
        self.evaluation_count += 1

    def get_sampling_probabilities(self):
        return self.weights / self.weights.sum()

    def get_metrics(self):
        probability = self.get_sampling_probabilities()
        return dict(emd_mean=self.emd.mean(), emd_max=self.emd.max(),
                    sampling_entropy=-(probability * probability.log()).sum(),
                    emd_evaluations=self.emd.new_tensor(self.evaluation_count))

    def state_dict(self):
        return dict(kind=self.kind, config=deepcopy(self.config), weights=self.weights.detach().cpu().clone(),
                    emd=self.emd.detach().cpu().clone(), last_evaluation_step=self.last_evaluation_step,
                    evaluation_count=self.evaluation_count)

    def validate_state_dict(self, state):
        if state.get("kind") != self.kind or state.get("config") != self.config:
            raise ValueError("EMD sampler checkpoint configuration differs")
        for key in ("weights", "emd"):
            value, target = state.get(key), getattr(self, key)
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                    or not torch.isfinite(value).all() or (value < 0).any()):
                raise ValueError(f"Invalid EMD sampler checkpoint {key}")
        step, count = state.get("last_evaluation_step"), state.get("evaluation_count")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (step, count)) or (not count and step):
            raise ValueError("Invalid EMD evaluation progress")
        expected = (torch.exp2(state["emd"].clamp(self.config["min_value"], self.config["max_value"]) * self.config["scale"])
                    if count else torch.ones_like(state["weights"]))
        if not torch.allclose(state["weights"], expected, rtol=2e-6, atol=0) or (not count and state["emd"].any()):
            raise ValueError("EMD checkpoint weights disagree with evaluation scores")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("weights", "emd"):
            getattr(self, key).copy_(state[key])
        self.last_evaluation_step, self.evaluation_count = state["last_evaluation_step"], state["evaluation_count"]
