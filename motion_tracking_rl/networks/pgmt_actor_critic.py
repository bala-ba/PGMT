"""PGMT attention with optional terrain K/V and PMT policy/normalization APIs.

The attention layout follows PGMT/pgmt/modules.py. Gaussian exploration,
normalization, initialization and checkpoint loading follow PMT ActorCritic.
History is supplied by the environment, so this is a feed-forward PPO policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from tensordict import TensorDict

from motion_tracking_rl.networks.actor_critic import ActorCritic
from motion_tracking_rl.networks.layers import EmpiricalNormalization, MLP
from motion_tracking_rl.networks.pgmt_terrain_glimpse import (
    PGMTTerrainCropCfg,
    PGMTTerrainGlimpse,
    PGMTTerrainGlimpseDebug,
    PGMTTerrainPatchEncoderCfg,
    PGMTTerrainSelectorCfg,
)
from motion_tracking_rl.pgmt_perception_contract import (
    terrain_observation_contract, validate_terrain_observation_contract,
)
from motion_tracking_rl.registry import register_network
from motion_tracking_rl.utils import build_obs_schema


class PGMTAttentionDebug(NamedTuple):
    """Independent, detached snapshots on the attention computation device.

    weights [N,heads,L,S] keeps each head; columns follow the input key order.
    key_valid [N,S] is boolean (True=participates). q_positions [L] and
    k_positions [S] are the applied RoPE positions, or None without RoPE.
    Weights reconstruct softmax from the actual rotated Q/K; fused SDPA can
    differ slightly due to backend rounding. Half inputs use float32 debug
    arithmetic. These values are for inspection, not auxiliary training loss.
    """

    weights: torch.Tensor
    key_valid: torch.Tensor
    q_positions: torch.Tensor | None
    k_positions: torch.Tensor | None


class PGMTAttentionOutput(NamedTuple):
    """Opt-in debug return; output retains its ordinary autograd graph."""

    output: torch.Tensor
    debug: PGMTAttentionDebug


class PGMTPerceptionDebug(NamedTuple):
    """Detached device-local snapshots; never cached or used as training inputs.

    intent [N,embedding_dim]; glimpse is None in disabled mode (otherwise contains
    centers in meters, normalized patch heights and validity/coverage).
    attention retains every head [N,heads,1,S] and the actual mask/RoPE.
    terrain_attention_mass [N,heads,1] sums columns 9:13, zero when disabled.
    key_labels describes each column; terrain_mode records the explicit mode.
    """

    intent: torch.Tensor
    glimpse: PGMTTerrainGlimpseDebug | None
    attention: PGMTAttentionDebug
    terrain_attention_mass: torch.Tensor
    key_labels: tuple[str, ...]
    terrain_mode: str


class PGMTCrossAttention(nn.Module):
    """Cross-attention with local PGMT RoPE positions and optional key masking.

    key_valid [N,S] is bool, True=allowed, matching PyTorch SDPA rather than
    MultiheadAttention's key_padding_mask convention. Inputs are finite tokens.
    A masked key contributes no attention weight or key/value input gradient.
    IFM must keep history/command keys valid, even when all terrain is invalid.

    For a generic all-keys-invalid row, SDPA produces zero attended features;
    the final output is out.bias, and debug weights are all zero. This is not
    the IFM's all-terrain-invalid case, which still has nine valid base keys.
    No learned parameters, persistent buffers or debug caches are added.
    """

    def __init__(self, dim: int, heads: int, rope_base: float | None = None):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or (rope_base is not None and (dim // heads) % 2):
            raise ValueError("Attention requires integral heads and an even RoPE head dimension")
        self.dim, self.heads, self.rope_base = dim, heads, rope_base
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def _split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose(1, 2)

    def _rotate(self, x, positions):
        frequency = self.rope_base ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
        angles = positions.to(x.dtype)[:, None] * frequency[None, :]
        c, s = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

    def forward(
        self, query, keys, *, q_positions=None, k_positions=None,
        key_valid: torch.Tensor | None = None, return_debug: bool = False,
    ) -> torch.Tensor | PGMTAttentionOutput:
        """Return [N,L,dim], or PGMTAttentionOutput when return_debug=True."""
        key_mask = None
        if key_valid is not None:
            if not isinstance(key_valid, torch.Tensor):
                raise TypeError("PGMT attention key_valid must be a Tensor")
            if key_valid.dtype != torch.bool:
                raise TypeError("PGMT attention key_valid must be boolean (True=valid)")
            if (query.ndim != 3 or keys.ndim != 3 or query.shape[0] != keys.shape[0]
                    or tuple(key_valid.shape) != tuple(keys.shape[:2])):
                raise ValueError("PGMT attention key_valid requires [N,S] matching query/key batch and key length")
            if key_valid.device != keys.device or key_valid.device != query.device:
                raise ValueError("PGMT attention key_valid, query and keys must share a device")
            key_mask = key_valid[:, None, None, :]
        q, k, v = self._split(self.q(query)), self._split(self.k(keys)), self._split(self.v(keys))
        if self.rope_base is not None:
            if q_positions is None or k_positions is None:
                raise ValueError("RoPE requires explicit query/key positions")
            q, k = self._rotate(q, q_positions), self._rotate(k, k_positions)
        # Preserve the previous SDPA call exactly when no mask is supplied.
        if key_mask is None:
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=0.0)
        output = self.out(attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.dim))
        if not return_debug:
            return output

        # Reuse the projected/rotated Q/K without retaining their training graph.
        # Avoid half-precision score overflow and autocast lowering debug matmul.
        with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
            scores = (q.detach().to(dtype) @ k.detach().to(dtype).transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            if key_mask is not None:
                scores.masked_fill_(~key_mask, float("-inf"))
                # An empty row has no distribution. Avoid softmax(-inf,...,-inf)
                # without a GPU->CPU branch; set its diagnostic weights to zero.
                scores = torch.where(key_mask.any(dim=-1, keepdim=True), scores, 0.0)
            weights = scores.softmax(dim=-1)
            if key_mask is not None:
                weights.masked_fill_(~key_mask, 0.0)
            valid = key_valid if key_valid is not None else torch.ones(
                keys.shape[:2], dtype=torch.bool, device=keys.device)
            debug = PGMTAttentionDebug(
                weights.detach().clone(), valid.detach().clone(),
                q_positions.detach().clone() if self.rope_base is not None else None,
                k_positions.detach().clone() if self.rope_base is not None else None,
            )
        return PGMTAttentionOutput(output, debug)


@register_network("PGMTActorCritic", compat_name="pgmt")
class PGMTActorCritic(ActorCritic):
    is_recurrent = False
    num_value_heads = 3
    value_groups = ("upper", "lower", "auxiliary")

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], num_actions: int,
        embedding_dim: int = 128, num_heads: int = 4,
        command_frames: int = 8, history_frames: int = 10, rope_base: float = 10000.0,
        actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
        activation: str = "elu", init_noise_std: float = 1.0,
        noise_std_type: str = "scalar", state_dependent_std: bool = False,
        actor_obs_normalization: bool = False, critic_obs_normalization: bool = False,
        forward_hidden_dims=None, forward_obs_normalization: bool | None = None,
        terrain_glimpse_cfg: dict | None = None, terrain_mode: str = "disabled",
        terrain_observation_cfg: dict | None = None,
        value_groups: tuple[str, ...] | list[str] = ("upper", "lower", "auxiliary"),
        **kwargs,
    ):
        from motion_tracking_rl.pgmt_value_contract import validate_value_groups
        self.value_groups = validate_value_groups(value_groups)
        self.num_value_heads = len(self.value_groups)
        if num_actions != 29 or command_frames != 8 or history_frames != 10:
            raise ValueError("PGMT stage 1 requires 29 actions, 8 command frames and 10 history frames")
        if rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if forward_hidden_dims is not None or forward_obs_normalization not in (None, False):
            raise ValueError("PGMT stage 1 does not use the native forward/residual MLP")
        if kwargs:
            raise TypeError(f"Unsupported PGMTActorCritic arguments: {sorted(kwargs)}")
        if terrain_mode not in ("disabled", "enabled"):
            raise ValueError("PGMT terrain_mode must be 'disabled' or 'enabled'")
        terrain_configs = None
        if terrain_glimpse_cfg is None:
            if terrain_mode != "disabled":
                raise ValueError("PGMT terrain_mode='enabled' requires terrain_glimpse_cfg (use {} for defaults)")
            if terrain_observation_cfg is not None:
                raise ValueError("PGMT terrain_observation_cfg requires terrain_glimpse_cfg")
        else:
            if not isinstance(terrain_glimpse_cfg, dict):
                raise TypeError("PGMT terrain_glimpse_cfg must be a dict or None")
            if embedding_dim != 128:
                raise ValueError("PGMT Terrain-Glimpse requires embedding_dim=128")
            config_types = dict(selector_config=PGMTTerrainSelectorCfg, crop_config=PGMTTerrainCropCfg,
                                encoder_config=PGMTTerrainPatchEncoderCfg)
            if terrain_glimpse_cfg.keys() - config_types.keys():
                raise TypeError("PGMT terrain_glimpse_cfg only accepts selector_config, crop_config, encoder_config")
            terrain_configs = {}
            for name, config_type in config_types.items():
                values = terrain_glimpse_cfg.get(name, {})
                if not isinstance(values, dict):
                    raise TypeError(f"PGMT terrain_glimpse_cfg.{name} must be a dict")
                terrain_configs[name] = config_type(**values)
            self._validate_terrain_obs_groups(obs_groups)
            terrain_observation_cfg = (terrain_observation_contract() if terrain_observation_cfg is None
                                       else deepcopy(terrain_observation_cfg))
            validate_terrain_observation_contract(terrain_observation_cfg)
        for group in ("policy", "critic", "policy_history", "command_window"):
            if not obs_groups.get(group):
                raise ValueError(f"PGMTActorCritic requires observation set '{group}'")
        self.history_frames, self.command_frames = history_frames, command_frames
        # Native ActorCritic creates the Gaussian parameters and raw-observation
        # normalizers. Replace only its MLP dimensions to include the intent token.
        super().__init__(
            obs, deepcopy(obs_groups), num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims,
            activation=activation, init_noise_std=init_noise_std,
            noise_std_type=noise_std_type, state_dependent_std=state_dependent_std,
        )
        proprio = self.get_actor_obs(obs)
        privileged = self.get_critic_obs(obs)
        if proprio.shape[-1] != 96:
            raise ValueError(f"PGMT proprioception must have 96 features, got {proprio.shape[-1]}")
        if terrain_mode == "enabled":
            self._terrain_observations(obs, proprio)
        self._sequence(obs, "policy_history", history_frames, 96)
        self._sequence(obs, "command_window", command_frames, 61)
        self.num_actions, self.privileged_dim = num_actions, privileged.shape[-1]
        self.proprio_embed = nn.Linear(96, embedding_dim)
        self.command_embed = nn.Linear(61, embedding_dim)
        self.history_attention = PGMTCrossAttention(embedding_dim, num_heads)
        self.intent_attention = PGMTCrossAttention(embedding_dim, num_heads, rope_base)
        self.actor = MLP(96 + embedding_dim, [2, num_actions] if state_dependent_std else num_actions,
                         actor_hidden_dims, activation)
        self.critic = MLP(self.privileged_dim + embedding_dim, self.num_value_heads, critic_hidden_dims, activation)
        print(f"PGMT: actor input={96 + embedding_dim}, critic input={self.privileged_dim + embedding_dim}, "
              f"value heads={self.num_value_heads} {self.value_groups}, embedding={embedding_dim}, "
              f"attention heads={num_heads}, terrain_mode={terrain_mode}")
        if state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            initial_std = init_noise_std if noise_std_type == "scalar" else torch.log(torch.tensor(init_noise_std + 1e-7))
            nn.init.constant_(self.actor[-2].bias[num_actions:], initial_std)
        self.command_obs_normalizer = EmpiricalNormalization(61) if actor_obs_normalization else nn.Identity()
        self.register_buffer("command_offsets", 2 ** torch.arange(command_frames) - 1, persistent=False)
        self._obs_schema = build_obs_schema(obs, self.obs_groups)
        self._terrain_observation_cfg = terrain_observation_cfg
        self._policy_kwargs = dict(
            embedding_dim=embedding_dim, num_heads=num_heads, command_frames=command_frames,
            history_frames=history_frames, rope_base=rope_base,
            actor_hidden_dims=list(actor_hidden_dims), critic_hidden_dims=list(critic_hidden_dims),
            activation=activation, init_noise_std=init_noise_std, noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std, actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
        )
        # Preserve historical three-head metadata exactly for strict P2 resumes.
        if self.num_value_heads == 4:
            self._policy_kwargs["value_groups"] = list(self.value_groups)
        # Construct AFTER all original layers: adding terrain must not change
        # their seeded initialization. None adds no parameters or buffers.
        self.terrain_glimpse = None if terrain_configs is None else PGMTTerrainGlimpse(**terrain_configs)
        self.terrain_mode = terrain_mode

    @property
    def terrain_mode(self) -> str:
        return self._terrain_mode

    @terrain_mode.setter
    def terrain_mode(self, mode: str):
        """Explicit regression switch; change only between rollout/update cycles."""
        if mode not in ("disabled", "enabled"):
            raise ValueError("PGMT terrain_mode must be 'disabled' or 'enabled'")
        if mode == "enabled" and self.terrain_glimpse is None:
            raise ValueError("PGMT terrain_mode='enabled' requires a constructed Terrain-Glimpse")
        self._terrain_mode = mode

    @staticmethod
    def _validate_terrain_obs_groups(obs_groups):
        """Two explicit independent 2-D inputs; no flattening into base groups."""
        keys = []
        for group in ("terrain_map", "terrain_valid"):
            mapping = obs_groups.get(group)
            if (not isinstance(mapping, (list, tuple)) or len(mapping) != 1
                    or not isinstance(mapping[0], str) or not mapping[0]):
                raise ValueError(f"PGMT obs_groups['{group}'] must name exactly one observation key")
            keys.append(mapping[0])
        if keys[0] == keys[1]:
            raise ValueError("PGMT terrain_map and terrain_valid must use distinct observation keys")
        for group in ("policy", "critic", "policy_history", "command_window"):
            if set(keys).intersection(obs_groups.get(group, [])):
                raise ValueError(f"PGMT terrain_map/terrain_valid must remain separate from '{group}'")

    def _terrain_observations(self, obs, reference):
        values = []
        for group in ("terrain_map", "terrain_valid"):
            key = self.obs_groups[group][0]
            if key not in obs:
                raise ValueError(f"PGMT enabled {group} requires observation '{key}'")
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"PGMT {group} must be a Tensor")
            if tuple(value.shape) != (reference.shape[0], 21, 21):
                raise ValueError(f"PGMT {group} requires [N,21,21] matching the proprio batch")
            if value.device != reference.device:
                raise ValueError(f"PGMT {group} device must match proprio")
            if group == "terrain_map":
                if not value.is_floating_point() or value.dtype != reference.dtype:
                    raise TypeError("PGMT terrain_map must share proprio's floating dtype")
            elif value.dtype != torch.bool and not value.is_floating_point():
                raise TypeError("PGMT terrain_valid must be boolean or floating point")
            values.append(value)
        return tuple(values)

    def _sequence(self, obs, group: str, frames: int, features: int):
        values = [obs[key] for key in self.obs_groups[group]]
        value = torch.cat(values, dim=-1)
        if value.ndim != 3 or value.shape[1:] != (frames, features):
            raise ValueError(f"PGMT {group} must have shape [N, {frames}, {features}], got {tuple(value.shape)}")
        return value

    def _encode(self, obs, *, return_debug: bool = False):
        """Shared actor/critic computation; maps are always saved raw observations.

        Normal calls return (proprio, intent) and make no diagnostic copies.
        Debug calls add a PGMTPerceptionDebug snapshot as the third result.
        There is deliberately no cross-forward token/graph cache.
        """
        proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
        history = self._sequence(obs, "policy_history", self.history_frames, 96)
        history = self.actor_obs_normalizer(history.reshape(-1, 96)).reshape_as(history)
        commands = self._sequence(obs, "command_window", self.command_frames, 61)
        commands = self.command_obs_normalizer(commands.reshape(-1, 61)).reshape_as(commands)
        token = self.history_attention(self.proprio_embed(proprio[:, None]), self.proprio_embed(history))
        keys = torch.cat((token, self.command_embed(commands)), dim=1)
        positions = torch.cat((self.command_offsets.new_zeros(1), self.command_offsets))
        key_valid, glimpse_debug = None, None
        if self.terrain_mode == "enabled":
            terrain_map, terrain_valid = self._terrain_observations(obs, proprio)
            # Pass the existing history token and already-normalized commands
            # directly. The glimpse input adapter alone normalizes the map.
            terrain = self.terrain_glimpse(terrain_map, terrain_valid, token, commands, return_debug=return_debug)
            glimpse_debug = terrain.debug
            key_valid = torch.cat((torch.ones(keys.shape[:2], device=keys.device, dtype=torch.bool),
                                   terrain.token_valid), dim=1)
            keys = torch.cat((keys, terrain.terrain_tokens), dim=1)
            positions = torch.cat((positions, positions.new_zeros(terrain.terrain_tokens.shape[1])))
        result = self.intent_attention(token, keys, q_positions=positions[:1], k_positions=positions,
                                       key_valid=key_valid, return_debug=return_debug)
        if return_debug:
            intent = result.output[:, 0]
            key_labels = ("history",) + tuple(f"command[{2**k - 1}]" for k in range(self.command_frames))
            if self.terrain_mode == "enabled":
                key_labels += tuple(f"terrain[{k}]" for k in range(4))
            debug = PGMTPerceptionDebug(
                intent.detach().clone(), glimpse_debug, result.debug,
                result.debug.weights[..., 9:].sum(-1).detach().clone(), key_labels, self.terrain_mode,
            )
            return proprio, intent, debug
        intent = result[:, 0]
        return proprio, intent

    @torch.no_grad()
    def get_perception_debug(self, obs) -> PGMTPerceptionDebug:
        """Recompute diagnostics without touching distribution/stats/RNG or mode.

        Uses the supplied observation pair, never requests new sensor samples.
        No .eval() toggle and no GPU->CPU copies; callers may transfer selected
        snapshots to CPU for display. Disabled/Stage-1 returns nine-key debug.
        """
        return self._encode(obs, return_debug=True)[2]

    def _update_distribution(self, obs):
        proprio, intent = self._encode(obs)
        self._update_distribution_from_encoding(proprio, intent)

    def _update_distribution_from_encoding(self, proprio, intent):
        output = self.actor(torch.cat((proprio, intent), dim=-1))
        if self.state_dependent_std:
            mean, scale = torch.unbind(output, dim=-2)
            std = scale if self.noise_std_type == "scalar" else scale.exp()
        else:
            mean = output
            std = (self.std if self.noise_std_type == "scalar" else self.log_std.exp()).expand_as(mean)
        # Exact native ActorCritic distribution guards, with no PGMT std clipping.
        std = torch.nan_to_num(std, nan=1e-3, posinf=1.0, neginf=1e-3).clamp_min(1e-6)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        self.distribution = Normal(mean, std)

    def act_and_evaluate(self, obs: TensorDict, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute values from one fresh shared encoding.

        PPO calls this once per rollout step or minibatch. Both losses retain
        their path into the same encoder graph; nothing is detached or cached
        across calls. Sampling is kept even during PPO recomputation, matching
        the RNG consumption of the standalone act()/evaluate() pair.
        """
        proprio, intent = self._encode(obs)
        self._update_distribution_from_encoding(proprio, intent)
        actions = self.distribution.sample()
        return actions, self._evaluate_from_encoding(obs, intent)

    def act_inference(self, obs):
        proprio, intent = self._encode(obs)
        output = self.actor(torch.cat((proprio, intent), dim=-1))
        return output[..., 0, :] if self.state_dependent_std else output

    def evaluate(self, obs, **kwargs):
        _, intent = self._encode(obs)
        return self._evaluate_from_encoding(obs, intent)

    def _evaluate_from_encoding(self, obs, intent):
        privileged = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(torch.cat((privileged, intent), dim=-1))

    def update_normalization(self, obs):
        super().update_normalization(obs)
        if self.actor_obs_normalization:
            commands = self._sequence(obs, "command_window", self.command_frames, 61)
            self.command_obs_normalizer.update(commands.reshape(-1, 61))

    def get_checkpoint_metadata(self):
        policy_kwargs = deepcopy(self._policy_kwargs)
        if self.terrain_glimpse is not None:
            policy_kwargs.update(terrain_mode=self.terrain_mode,
                                 terrain_observation_cfg=deepcopy(self._terrain_observation_cfg), terrain_glimpse_cfg=dict(
                selector_config=asdict(self.terrain_glimpse.selector.config),
                crop_config=asdict(self.terrain_glimpse.crop.config),
                encoder_config=asdict(self.terrain_glimpse.patch_encoder.config),
            ))
        metadata = {
            "policy_class": type(self).__name__,
            "policy_family": ("pgmt_terrain_tracking" if self.num_value_heads == 4 else
                              "pgmt_stage1" if self.terrain_glimpse is None else "pgmt_perception_only"),
            "policy_kwargs": policy_kwargs, "obs_groups": deepcopy(self.obs_groups),
            "obs_schema": deepcopy(self._obs_schema), "value_groups": list(self.value_groups),
            "num_actions": self.num_actions,
            "action_contract": "q_target = q_default + G1_ACTION_SCALE * action",
            "signature": {"num_actions": self.num_actions, "privileged_dim": self.privileged_dim,
                          "num_value_heads": self.num_value_heads, **policy_kwargs},
        }
        if self.terrain_glimpse is not None:
            contract = dict(
                observation=deepcopy(self._terrain_observation_cfg),
                obs_groups={name: list(self.obs_groups[name]) for name in ("terrain_map", "terrain_valid")},
                network_input=dict(shape=[21, 21], units="m", mask_dtype="bool or floating (>0.5 valid)",
                                   preprocessing="clip(-3,3)/3 once per _encode; no augmentation or running stats"),
                fusion=dict(query="history", key_order=["history", "command*8", "terrain*4"],
                            query_rope_position=0, key_rope_positions=[0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0],
                            key_valid="[True]*9 + patch_valid", terrain_logit_bias=0.0,
                            # Historical standalone API contract: act/evaluate
                            # still encode independently. PPO's joint execution
                            # is recorded separately in execution_options so old
                            # checkpoints retain strict semantic compatibility.
                            actor_critic="shared modules and saved augmented map/mask; separate forward calls"),
            )
            metadata["perception_contract"] = contract
            metadata["signature"].update(perception_contract=deepcopy(contract), value_groups=list(self.value_groups))
        return metadata
