"""Terrain-Glimpse with shared patch encoding, also usable standalone.

PGMTActorCritic optionally consumes its tokens/mask in the P2.3 IFM path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from motion_tracking_rl.networks.layers import MLP
from motion_tracking_rl.networks.pgmt_terrain_input import PGMTTerrainInput


class PGMTTerrainGlimpseInputs(NamedTuple):
    """Prepared inputs; map heights are normalized, not meters.

    ``terrain_map`` and ``terrain_valid`` are [N,21,21], row=y/col=x in
    actual torso-yaw coordinates. Invalid height sentinels are preserved.
    ``history_token`` [N,1,128] and ``commands`` [N,8,61] retain the caller's
    tensors and autograd graphs. Treat all returned tensors as read-only.
    """

    terrain_map: Tensor
    terrain_valid: Tensor
    history_token: Tensor
    commands: Tensor


class PGMTTerrainGlimpseInput(nn.Module):
    """Prepare version-3 observations for the Terrain-Glimpse encoder.

    Inputs:
        terrain_map: [N,21,21], finite metric sensor_z - ground_z heights
            from P2.1, after PMT fill/dropout/noise/clipping.
        terrain_valid: [N,21,21], boolean or floating mask from the *same*
            observation call; includes missing hits, bounds and dropout.
        history_token: [N,1,128], existing history-attention output.
        commands: [N,8,61], existing command-normalizer output, before
            command embedding. This module does not normalize it again.

    Map, history and commands must share a floating dtype; all inputs must
    share a device and batch size. The mask is converted to the map dtype by
    PGMTTerrainInput. There are no implicit device transfers, learned weights,
    running statistics or random draws. Feed the saved observation pair during
    PPO recomputation rather than requesting a new environment observation.

    Only the map is normalized here, using PMT's clip(-3,3)/3. In particular,
    invalid heights are not zeroed or replaced with a new constant. The caller
    supplies finite version-3 observations; no additional sensor repair or
    augmentation is performed by this network input adapter.
    """

    def __init__(self):
        super().__init__()
        self.terrain_input = PGMTTerrainInput(z_clip=3.0, normalize_height=True)

    def forward(
        self, terrain_map: Tensor, terrain_valid: Tensor,
        history_token: Tensor, commands: Tensor,
    ) -> PGMTTerrainGlimpseInputs:
        inputs = (
            ("terrain_map", terrain_map, (21, 21)),
            ("terrain_valid", terrain_valid, (21, 21)),
            ("history_token", history_token, (1, 128)),
            ("commands", commands, (8, 61)),
        )
        for name, value, shape in inputs:
            if not isinstance(value, Tensor):
                raise TypeError(f"PGMT Terrain-Glimpse {name} must be a Tensor")
            if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
                raise ValueError(f"PGMT Terrain-Glimpse {name} requires [N,{','.join(map(str, shape))}], "
                                 f"got {tuple(value.shape)}")
            if value.shape[0] != terrain_map.shape[0]:
                raise ValueError(f"PGMT Terrain-Glimpse {name} batch size must match terrain_map")
            if value.device != terrain_map.device:
                raise ValueError(f"PGMT Terrain-Glimpse {name} device must match terrain_map")
            if name == "terrain_valid":
                if value.dtype != torch.bool and not value.is_floating_point():
                    raise TypeError("PGMT Terrain-Glimpse terrain_valid must be boolean or floating point")
            elif not value.is_floating_point() or value.dtype != terrain_map.dtype:
                raise TypeError(f"PGMT Terrain-Glimpse {name} must share terrain_map's floating dtype")

        height, valid = self.terrain_input(terrain_map, terrain_valid)
        return PGMTTerrainGlimpseInputs(height, valid, history_token, commands)


@dataclass(frozen=True)
class PGMTTerrainSelectorCfg:
    """Local engineering choices, not hyperparameters disclosed by the paper.

    Four centers are fixed by the current PGMT contract. The 21x21 map spans
    +/-1 m; the 5x5 crop spans +/-0.2 m about each center. Therefore
    ``center_limit_m`` may be reduced but must never exceed 0.8 m.
    """

    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    center_limit_m: float = 0.8

    def __post_init__(self):
        if (not isinstance(self.hidden_dims, (tuple, list)) or not self.hidden_dims
                or any(type(width) is not int or width <= 0 for width in self.hidden_dims)):
            raise ValueError("PGMT selector hidden_dims must contain positive integers")
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if (isinstance(self.center_limit_m, bool) or not isinstance(self.center_limit_m, (int, float))
                or not math.isfinite(self.center_limit_m) or not 0 < self.center_limit_m <= 0.8):
            raise ValueError("PGMT selector center_limit_m must be finite and in (0, 0.8]")
        if not isinstance(self.activation, str):
            raise TypeError("PGMT selector activation must be a name supported by the PMT MLP")


class PGMTTerrainPositionSelector(nn.Module):
    """Predict four (x,y) centers in meters in the actual torso-yaw frame.

    Accept only the prepared output of PGMTTerrainGlimpseInput. Concatenation
    order is height[441], validity[441], history[128], commands[8*61]; maps
    flatten row=y then col=x, and commands flatten frame then feature.
    There is no second normalization, augmentation or detached input cache.

    The default MLP is 1498 -> 256 -> 128 -> 8 with ELU hidden activations and
    native Linear initialization. Independent output rows allow four distinct
    centers; they are not fixed footholds and no diversity loss is imposed.
    An all-invalid *finite* map still yields bounded centers. Patch validity
    comes from PGMTMaskedTerrainCrop; attention masking belongs to the IFM.
    """

    input_dim = 441 + 441 + 128 + 8 * 61
    num_glimpses = 4

    def __init__(self, config: PGMTTerrainSelectorCfg | None = None):
        super().__init__()
        if config is not None and not isinstance(config, PGMTTerrainSelectorCfg):
            raise TypeError("PGMT selector config must be PGMTTerrainSelectorCfg")
        self.config = PGMTTerrainSelectorCfg() if config is None else config
        self.mlp = MLP(self.input_dim, self.num_glimpses * 2,
                       self.config.hidden_dims, self.config.activation)

    def forward(self, prepared: PGMTTerrainGlimpseInputs) -> Tensor:
        """Return [N,4,2] center_xy_m, retaining gradients through all inputs."""
        if not isinstance(prepared, PGMTTerrainGlimpseInputs):
            raise TypeError("PGMT selector requires PGMTTerrainGlimpseInputs from PGMTTerrainGlimpseInput")
        features = torch.cat((prepared.terrain_map.flatten(1), prepared.terrain_valid.flatten(1),
                              prepared.history_token.flatten(1), prepared.commands.flatten(1)), dim=-1)
        raw = self.mlp(features).reshape(features.shape[0], self.num_glimpses, 2)
        return self.config.center_limit_m * raw.tanh()


@dataclass(frozen=True)
class PGMTTerrainCropCfg:
    """Mask thresholds are engineering choices for PMT's dropped scan points.

    Geometry is fixed by the current input/selector contract: 21x21 over 2 m,
    four crops of 5x5 points at 0.1 m spacing. Output invalid height is -1 in
    network-normalized units. This does not overwrite the source observation.
    """

    min_valid_weight: float = 0.5
    min_valid_points: int = 13
    weight_epsilon: float = 1e-6

    def __post_init__(self):
        if (isinstance(self.min_valid_weight, bool) or not isinstance(self.min_valid_weight, (int, float))
                or not math.isfinite(self.min_valid_weight) or not 0 < self.min_valid_weight <= 1):
            raise ValueError("PGMT crop min_valid_weight must be finite and in (0, 1]")
        if type(self.min_valid_points) is not int or not 1 <= self.min_valid_points <= 25:
            raise ValueError("PGMT crop min_valid_points must be an integer in [1, 25]")
        if (isinstance(self.weight_epsilon, bool) or not isinstance(self.weight_epsilon, (int, float))
                or not math.isfinite(self.weight_epsilon) or not 0 < self.weight_epsilon <= self.min_valid_weight):
            raise ValueError("PGMT crop weight_epsilon must be finite and in (0, min_valid_weight]")


class PGMTTerrainPatches(NamedTuple):
    """Crop results; retain autograd graphs and treat tensors as read-only.

    height [N,4,5,5]: normalized heights, invalid points filled with -1.
    point_valid [N,4,5,5]: boolean, valid interpolation weight >= threshold.
    valid_weight [N,4,5,5]: bilinear valid support, zero outside the scan.
    coverage [N,4]: fraction of valid sampled points (not mean support weight).
    patch_valid [N,4]: boolean, valid-point count >= configured minimum.

    Patch rows are local y, columns local x. A rejected patch retains its
    per-point values for diagnostics; patch_valid must mask its future token.
    """

    height: Tensor
    point_valid: Tensor
    valid_weight: Tensor
    coverage: Tensor
    patch_valid: Tensor


class PGMTMaskedTerrainCrop(nn.Module):
    """Differentiable 5x5 crops with missing-data-aware bilinear interpolation.

    Both arguments come from this module's preceding stages: prepared maps
    are already normalized, center_xy_m [N,4,2] is in actual torso-yaw meters.
    Numerator and denominator are sampled together on the SAME grid, using
    zero padding and align_corners=True. No map normalization, noise, border
    replication, coordinate clamping or CPU observation copies occur here.

    Heights/weights are differentiable within interpolation cells. Validity
    decisions are discrete; below-threshold samples have zero height gradient.
    Non-finite source heights are excluded from support (including NaN behind
    mask=0); non-finite/out-of-scan sample coordinates return invalid points.
    These local guards do not replace the version-3 input contract upstream.
    """

    def __init__(self, config: PGMTTerrainCropCfg | None = None):
        super().__init__()
        if config is not None and not isinstance(config, PGMTTerrainCropCfg):
            raise TypeError("PGMT crop config must be PGMTTerrainCropCfg")
        self.config = PGMTTerrainCropCfg() if config is None else config
        # Integer offsets avoid baking float32 rounding into double-precision
        # derivative checks. This small geometry buffer follows module.to().
        dy, dx = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing="ij")
        self.register_buffer("_offset_grid", torch.stack((dx, dy), dim=-1), persistent=False)

    def forward(self, prepared: PGMTTerrainGlimpseInputs, center_xy_m: Tensor) -> PGMTTerrainPatches:
        if not isinstance(prepared, PGMTTerrainGlimpseInputs):
            raise TypeError("PGMT crop requires PGMTTerrainGlimpseInputs from PGMTTerrainGlimpseInput")
        height, valid = prepared.terrain_map, prepared.terrain_valid
        if height.ndim != 3 or tuple(height.shape[1:]) != (21, 21) or valid.shape != height.shape:
            raise ValueError("PGMT crop requires matching map/mask [N,21,21]")
        if not isinstance(center_xy_m, Tensor):
            raise TypeError("PGMT crop center_xy_m must be a Tensor")
        if tuple(center_xy_m.shape) != (height.shape[0], 4, 2):
            raise ValueError("PGMT crop center_xy_m requires [N,4,2] with the map batch size")
        if not height.is_floating_point() or center_xy_m.dtype != height.dtype:
            raise TypeError("PGMT crop centers must share the map's floating dtype")
        if any(value.device != height.device for value in (valid, center_xy_m, self._offset_grid)):
            raise ValueError("PGMT crop map, mask, centers and module must share a device")

        offsets_m = self._offset_grid.to(dtype=height.dtype) * 0.1
        sample_xy_m = center_xy_m[:, :, None, None, :] + offsets_m
        # Map midpoint is (0,0). Convert meters to normalized grid coordinates:
        # grid = 2 * (xy - midpoint) / scan_size_m, with scan_size_m=(2,2).
        grid = 2.0 * sample_xy_m / 2.0
        in_scan = torch.isfinite(grid).all(dim=-1) & (grid.abs() <= 1.0).all(dim=-1)
        # grid_sample treats NaN coordinates specially; explicitly invalidate
        # them instead, and keep all intermediate arithmetic finite.
        safe_grid = torch.where(in_scan[..., None], grid, 0.0)
        source_valid = torch.isfinite(valid) & (valid > 0.5) & torch.isfinite(height)
        # Unlike height * mask, where excludes NaN/Inf at invalid source points.
        source_height = torch.where(source_valid, height, 0.0)
        source = torch.stack((source_height, source_valid.to(height.dtype)), dim=1)
        n = height.shape[0]
        # Stack the four crop grids vertically, keeping only one map per env.
        sampled = F.grid_sample(source, safe_grid.reshape(n, 20, 5, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=True).reshape(n, 2, 4, 5, 5)
        numerator = torch.where(in_scan, sampled[:, 0], 0.0)
        weight = torch.where(in_scan, sampled[:, 1].clamp(0.0, 1.0), 0.0)
        point_valid = in_scan & (weight >= self.config.min_valid_weight)
        interpolated = numerator / weight.clamp_min(self.config.weight_epsilon)
        patch_height = torch.where(point_valid, interpolated, -1.0)
        valid_count = point_valid.sum(dim=(-2, -1))
        coverage = valid_count.to(height.dtype) / 25.0
        patch_valid = valid_count >= self.config.min_valid_points
        return PGMTTerrainPatches(patch_height, point_valid, weight, coverage, patch_valid)


@dataclass(frozen=True)
class PGMTTerrainPatchEncoderCfg:
    """Shared MLP widths/activation are engineering choices, as in the selector.

    Input width 52 and output width 128 are fixed by the current PGMT contract.
    """

    hidden_dims: tuple[int, ...] = (128,)
    activation: str = "elu"

    def __post_init__(self):
        if (not isinstance(self.hidden_dims, (tuple, list)) or not self.hidden_dims
                or any(type(width) is not int or width <= 0 for width in self.hidden_dims)):
            raise ValueError("PGMT patch encoder hidden_dims must contain positive integers")
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not isinstance(self.activation, str):
            raise TypeError("PGMT patch encoder activation must be a name supported by the PMT MLP")


class PGMTTerrainPatchEncoder(nn.Module):
    """Encode all four patches with ONE shared 52 -> 128 -> 128 MLP.

    Features are normalized height[25], boolean point validity cast to the
    height dtype[25], then center (x,y)[2] in torso-yaw meters. Patch flattening
    preserves row=y/col=x. No additional normalization or augmentation occurs.

    Linear layers operate on the last axis of [N,4,52], sharing their weights
    across environments and patch indices. Heights and centers keep autograd.
    Invalid patches still yield finite embeddings for finite inputs, possibly
    nonzero due to MLP biases. Consumers MUST carry patches.patch_valid into
    attention masking; a zero vector alone would not exclude a softmax entry.
    """

    input_dim = 25 + 25 + 2
    token_dim = 128

    def __init__(self, config: PGMTTerrainPatchEncoderCfg | None = None):
        super().__init__()
        if config is not None and not isinstance(config, PGMTTerrainPatchEncoderCfg):
            raise TypeError("PGMT patch encoder config must be PGMTTerrainPatchEncoderCfg")
        self.config = PGMTTerrainPatchEncoderCfg() if config is None else config
        self.mlp = MLP(self.input_dim, self.token_dim, self.config.hidden_dims, self.config.activation)

    def forward(self, patches: PGMTTerrainPatches, center_xy_m: Tensor) -> Tensor:
        """Return terrain_tokens [N,4,128]; no patch is silently discarded."""
        if not isinstance(patches, PGMTTerrainPatches):
            raise TypeError("PGMT patch encoder requires PGMTTerrainPatches from PGMTMaskedTerrainCrop")
        height, point_valid = patches.height, patches.point_valid
        if height.ndim != 4 or tuple(height.shape[1:]) != (4, 5, 5) or point_valid.shape != height.shape:
            raise ValueError("PGMT patch encoder requires matching height/point_valid [N,4,5,5]")
        if not isinstance(center_xy_m, Tensor):
            raise TypeError("PGMT patch encoder center_xy_m must be a Tensor")
        if tuple(center_xy_m.shape) != (height.shape[0], 4, 2):
            raise ValueError("PGMT patch encoder center_xy_m requires [N,4,2] with the patch batch size")
        if not height.is_floating_point() or center_xy_m.dtype != height.dtype:
            raise TypeError("PGMT patch encoder centers must share the height's floating dtype")
        if point_valid.dtype != torch.bool:
            raise TypeError("PGMT patch encoder point_valid must be boolean")
        if any(value.device != height.device for value in (point_valid, center_xy_m)):
            raise ValueError("PGMT patch encoder heights, point_valid and centers must share a device")
        features = torch.cat((height.flatten(-2), point_valid.to(height.dtype).flatten(-2),
                              center_xy_m), dim=-1)
        return self.mlp(features)


class PGMTTerrainGlimpseDebug(NamedTuple):
    """Optional detached, independent snapshots on the computation device.

    center_xy_m [N,4,2] is in actual torso-yaw meters; patch_height [N,4,5,5]
    is NORMALIZED sensor-ground height (multiply by 3 for meters). Invalid
    heights are -1. point_valid and valid_weight are [N,4,5,5], coverage and
    token_valid are [N,4]. Coverage counts valid sampled points / 25.
    These copies never feed the encoder or future IFM. Display code may move
    selected environments to CPU without retaining the training graph.
    """

    center_xy_m: Tensor
    patch_height: Tensor
    point_valid: Tensor
    valid_weight: Tensor
    coverage: Tensor
    token_valid: Tensor


class PGMTTerrainGlimpseOutput(NamedTuple):
    """Trainable tokens plus boolean validity; optional debug is display-only.

    terrain_tokens [N,4,128] retains autograd. token_valid [N,4] is exactly
    crop.patch_valid and must mask terrain K/V in P2.3, including the case
    where all four patches are invalid. Treat the main outputs as read-only.
    """

    terrain_tokens: Tensor
    token_valid: Tensor
    debug: PGMTTerrainGlimpseDebug | None


class PGMTTerrainGlimpse(nn.Module):
    """Compose input -> selector -> masked crop -> shared patch encoder.

    The four input arguments follow PGMTTerrainGlimpseInput's version-3
    contract: raw *metric* map/mask and existing history/normalized commands.
    Map preprocessing runs exactly once per forward. No RNG draws, observation
    resampling, implicit device copies or persistent debug caches are added.
    Module construction performs the native PMT MLP parameter initialization.

    Debug snapshots are opt-in to avoid per-step copies during training. Their
    detach().clone() does not sever or alias the main computation graph.
    The caller owns IFM fusion; the default Stage-1 policy remains blind.
    """

    def __init__(
        self, selector_config: PGMTTerrainSelectorCfg | None = None,
        crop_config: PGMTTerrainCropCfg | None = None,
        encoder_config: PGMTTerrainPatchEncoderCfg | None = None,
    ):
        super().__init__()
        self.input_adapter = PGMTTerrainGlimpseInput()
        self.selector = PGMTTerrainPositionSelector(selector_config)
        self.crop = PGMTMaskedTerrainCrop(crop_config)
        self.patch_encoder = PGMTTerrainPatchEncoder(encoder_config)

    def forward(
        self, terrain_map: Tensor, terrain_valid: Tensor,
        history_token: Tensor, commands: Tensor, *, return_debug: bool = False,
    ) -> PGMTTerrainGlimpseOutput:
        prepared = self.input_adapter(terrain_map, terrain_valid, history_token, commands)
        centers = self.selector(prepared)
        patches = self.crop(prepared, centers)
        tokens = self.patch_encoder(patches, centers)
        debug = None
        if return_debug:
            debug = PGMTTerrainGlimpseDebug(*(value.detach().clone() for value in (
                centers, patches.height, patches.point_valid, patches.valid_weight,
                patches.coverage, patches.patch_valid,
            )))
        return PGMTTerrainGlimpseOutput(tokens, patches.patch_valid, debug)
