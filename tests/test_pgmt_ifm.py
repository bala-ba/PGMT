"""P2.3 attention masking and Terrain-Glimpse integration contracts."""
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from tensordict import TensorDict

from motion_tracking_rl.networks.pgmt_actor_critic import (
    PGMTActorCritic, PGMTCrossAttention, PGMTAttentionOutput, PGMTPerceptionDebug,
)
from motion_tracking_rl.pgmt_perception_contract import terrain_observation_contract


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(device, *, length=1, dim=128, dtype=torch.float32, rope=True):
    torch.manual_seed(2311)
    module = PGMTCrossAttention(dim, 4, 10000. if rope else None).to(device=device, dtype=dtype)
    query = torch.randn(4, length, dim, device=device, dtype=dtype, requires_grad=True)
    keys = torch.randn(4, 13, dim, device=device, dtype=dtype, requires_grad=True)
    # Mixed rows: all valid, two terrain keys, no terrain, no keys at all.
    valid = torch.ones(4, 13, device=device, dtype=torch.bool)
    valid[1, [10, 12]] = False
    valid[2, 9:] = False
    valid[3] = False
    positions = dict(q_positions=torch.arange(length, device=device),
                     k_positions=torch.tensor([0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0], device=device))
    return module, query, keys, valid, positions


def numpy_attention(module, query, keys, valid, positions):
    """Independent float64 score/softmax oracle; complex multiplication for RoPE."""
    def array(value):
        return value.detach().cpu().numpy().astype(np.float64)

    def project(value, linear):
        return (array(value) @ array(linear.weight).T + array(linear.bias)).reshape(
            value.shape[0], value.shape[1], module.heads, -1).transpose(0, 2, 1, 3)

    def rotate(value, token_positions):
        angle = array(token_positions)[:, None] * module.rope_base ** (
            -np.arange(0, value.shape[-1], 2) / value.shape[-1])
        rotated = (value[..., ::2] + 1j * value[..., 1::2]) * np.exp(1j * angle)
        return np.stack((rotated.real, rotated.imag), -1).reshape(value.shape)

    q, k, v = project(query, module.q), project(keys, module.k), project(keys, module.v)
    if module.rope_base is not None:
        q, k = rotate(q, positions["q_positions"]), rotate(k, positions["k_positions"])
    mask = valid.cpu().numpy()
    weights = np.zeros((query.shape[0], module.heads, query.shape[1], keys.shape[1]))
    features = np.zeros_like(q)
    for env in range(query.shape[0]):
        selected = np.flatnonzero(mask[env])
        if not len(selected):
            continue
        for head in range(module.heads):
            scores = q[env, head] @ k[env, head, selected].T / math.sqrt(q.shape[-1])
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            weights[env, head][:, selected] = probs
            features[env, head] = probs @ v[env, head, selected]
    features = features.transpose(0, 2, 1, 3).reshape(query.shape)
    output = features @ array(module.out.weight).T + array(module.out.bias)
    return output, weights


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope", [False, True])
def test_unmasked_attention_preserves_original_sdpa_call_and_state(device, rope):
    module, query, keys, _, positions = sample(device, rope=rope)
    q, k, v = module._split(module.q(query)), module._split(module.k(keys)), module._split(module.v(keys))
    if rope:
        q, k = module._rotate(q, positions["q_positions"]), module._rotate(k, positions["k_positions"])
    expected = module.out(F.scaled_dot_product_attention(q, k, v, dropout_p=0.).transpose(1, 2).reshape_as(query))
    actual = module(query, keys, **positions)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert set(module.state_dict()) == {f"{layer}.{part}" for layer in ("q", "k", "v", "out")
                                      for part in ("weight", "bias")}
    all_valid = module(query, keys, **positions, key_valid=torch.ones(4, 13, device=device, dtype=torch.bool))
    torch.testing.assert_close(all_valid, actual, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("length", [1, 3])
def test_masked_attention_and_per_head_debug_match_independent_numpy(device, rope, length):
    module, query, keys, valid, positions = sample(device, length=length, dim=16, dtype=torch.float64, rope=rope)
    result = module(query, keys, **positions, key_valid=valid, return_debug=True)
    assert isinstance(result, PGMTAttentionOutput)
    expected, expected_weights = numpy_attention(module, query, keys, valid, positions)
    np.testing.assert_allclose(result.output.detach().cpu(), expected, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(result.debug.weights.cpu(), expected_weights, atol=1e-12, rtol=1e-11)
    assert result.debug.weights.shape == (4, 4, length, 13)
    masked = ~valid[:, None, None, :].expand_as(result.debug.weights)
    assert result.debug.weights[masked].eq(0).all()
    expected_sum = valid.any(-1).to(torch.float64)[:, None, None].expand(4, 4, length)
    torch.testing.assert_close(result.debug.weights.sum(-1), expected_sum, atol=1e-14, rtol=1e-14)
    assert not torch.equal(result.debug.weights[0, 0], result.debug.weights[0, 1])
    torch.testing.assert_close(result.output[3], module.out.bias.expand_as(result.output[3]))


@pytest.mark.parametrize("device", DEVICES)
def test_masked_key_values_and_gradients_are_excluded_with_nine_key_fallback(device):
    module, query, keys, valid, positions = sample(device)
    result = module(query, keys, **positions, key_valid=valid)
    modified = keys.detach().clone()
    modified[~valid] = 1000. + torch.arange(128, device=device)  # Finite, conspicuously different tokens.
    changed = module(query, modified, **positions, key_valid=valid)
    torch.testing.assert_close(changed, result, atol=1e-6, rtol=1e-5)
    # All four terrain keys invalid: compare to physically removing them.
    base = module(query[2:3], keys[2:3, :9], q_positions=positions["q_positions"],
                  k_positions=positions["k_positions"][:9])
    torch.testing.assert_close(result[2:3], base, atol=1e-6, rtol=1e-5)
    result.square().sum().backward()
    assert keys.grad[~valid].eq(0).all()
    assert query.grad[3].eq(0).all()
    assert torch.isfinite(keys.grad).all() and keys.grad[valid].abs().sum() > 0
    assert torch.isfinite(query.grad).all() and query.grad[:3].abs().sum() > 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("masked", [False, True])
def test_debug_snapshots_do_not_alias_inputs_or_change_output_gradients_rng(device, masked):
    module, query, keys, valid, positions = sample(device)
    kwargs = {**positions, "key_valid": valid if masked else None}
    snapshots = [t.detach().clone() for t in (query, keys, valid, *positions.values())]
    rng = (torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()).clone()
    state_before = {name: p.detach().clone() for name, p in module.state_dict().items()}
    plain = module(query, keys, **kwargs)
    result = module(query, keys, **kwargs, return_debug=True)
    torch.testing.assert_close(result.output, plain, atol=0, rtol=0)
    for name, tensor in result.debug._asdict().items():
        assert tensor.grad_fn is None and not tensor.requires_grad
        if name in positions:
            torch.testing.assert_close(tensor, positions[name], atol=0, rtol=0)
            assert tensor.data_ptr() != positions[name].data_ptr()
        elif name == "key_valid":
            torch.testing.assert_close(tensor, valid if masked else torch.ones_like(valid))
            assert tensor.data_ptr() != valid.data_ptr()
        tensor.zero_()  # Display-side changes must not corrupt saved tensors for backward.
    targets = (query, keys, *tuple(module.parameters()))
    a = torch.autograd.grad(plain.square().mean(), targets)
    b = torch.autograd.grad(result.output.square().mean(), targets)
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, atol=1e-7, rtol=1e-5)
    for value, before in zip((query, keys, valid, *positions.values()), snapshots):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
    module.eval()
    with torch.no_grad():
        torch.testing.assert_close(module(query, keys, **kwargs), plain, atol=0, rtol=0)
    after_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(rng, after_rng)
    assert all(torch.equal(value, state_before[name]) for name, value in module.state_dict().items())


@pytest.mark.parametrize("device", DEVICES)
def test_all_keys_invalid_has_finite_bias_only_output_and_zero_key_query_gradient(device):
    module, query, keys, valid, positions = sample(device)
    valid.zero_()
    result = module(query, keys, **positions, key_valid=valid, return_debug=True)
    torch.testing.assert_close(result.output, module.out.bias.expand_as(result.output), rtol=0, atol=0)
    assert result.debug.weights.eq(0).all()
    result.output.sum().backward()
    assert query.grad.eq(0).all() and keys.grad.eq(0).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if name != "out.bias":
            assert parameter.grad.eq(0).all(), name


@pytest.mark.parametrize("problem", ["not_tensor", "float", "integer", "batch", "length", "broadcast_shape", "device"])
def test_key_mask_rejects_ambiguous_or_incompatible_inputs(problem):
    module, query, keys, valid, positions = sample("cpu")
    bad = {"not_tensor": [[True] * 13] * 4, "float": valid.float(), "integer": valid.long(),
           "batch": valid[:1], "length": valid[:, :9], "broadcast_shape": valid[:, None, None],
           "device": valid.to("meta")}[problem]
    with pytest.raises((ValueError, TypeError), match="PGMT attention key_valid"):
        module(query, keys, **positions, key_valid=bad)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_ifm_layout_matches_numpy_and_debug_records_applied_positions(device):
    module, query, keys, valid, positions = sample(device)
    result = module(query, keys, **positions, key_valid=valid, return_debug=True)
    expected, weights = numpy_attention(module, query, keys, valid, positions)
    np.testing.assert_allclose(result.output.detach().cpu(), expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(result.debug.weights.cpu(), weights, atol=1e-6, rtol=1e-5)
    assert result.debug.k_positions.tolist() == [0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0]
    no_rope = PGMTCrossAttention(128, 4).to(device)
    debug = no_rope(query, keys, key_valid=valid, return_debug=True).debug
    assert debug.q_positions is None and debug.k_positions is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_debug_uses_finite_float32_scores(dtype):
    module, query, keys, valid, positions = sample("cuda")
    # Scores exceed float16 range; SDPA and float32 debug must remain finite.
    with torch.no_grad():
        module.q.weight.copy_(torch.eye(128, device="cuda"))
        module.k.weight.copy_(torch.eye(128, device="cuda"))
        module.q.bias.zero_()
        module.k.bias.zero_()
        query.mul_(500.)
        keys.mul_(500.)
    with torch.autocast("cuda", dtype=dtype):
        plain = module(query, keys, **positions, key_valid=valid)
        result = module(query, keys, **positions, key_valid=valid, return_debug=True)
    assert result.debug.weights.dtype == torch.float32
    assert torch.isfinite(result.debug.weights).all() and torch.isfinite(result.output).all()
    torch.testing.assert_close(result.output, plain, atol=0, rtol=0)
    torch.testing.assert_close(result.debug.weights.sum(-1), valid.any(-1)[:, None, None].float().expand(4, 4, 1))


POLICY_GROUPS = {"policy": ["proprio"], "critic": ["critic"],
                 "policy_history": ["history"], "command_window": ["commands"]}


def terrain_observations(device, n=3):
    torch.manual_seed(2322)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 21, device=device),
                          torch.linspace(-1, 1, 21, device=device), indexing="ij")
    return TensorDict(dict(proprio=torch.randn(n, 96, device=device),
                           critic=torch.randn(n, 251, device=device),
                           history=torch.randn(n, 10, 96, device=device),
                           commands=torch.randn(n, 8, 61, device=device),
                           terrain_map=(.8 + .7*x + .3*y + .2*x*y).expand(n, -1, -1).clone(),
                           terrain_valid=torch.ones(n, 21, 21, device=device, dtype=torch.bool)), [n])


def make_policy(obs, **kwargs):
    groups = dict(POLICY_GROUPS)
    if kwargs.get("terrain_glimpse_cfg") is not None:
        groups.update(terrain_map=["terrain_map"], terrain_valid=["terrain_valid"])
    return PGMTActorCritic(obs, groups, 29, actor_hidden_dims=[32, 16],
                          critic_hidden_dims=[32, 16], **kwargs).to(obs["proprio"].device)


def legacy_encode(model, obs):
    """Nine-key policy path from before terrain integration."""
    proprio = model.actor_obs_normalizer(obs["proprio"])
    history = model.actor_obs_normalizer(obs["history"].flatten(0, 1)).reshape(-1, 10, 96)
    commands = model.command_obs_normalizer(obs["commands"].flatten(0, 1)).reshape(-1, 8, 61)
    token = model.history_attention(model.proprio_embed(proprio[:, None]), model.proprio_embed(history))
    keys = torch.cat((token, model.command_embed(commands)), 1)
    positions = torch.tensor([0, 0, 1, 3, 7, 15, 31, 63, 127], device=keys.device)
    return proprio, model.intent_attention(token, keys, q_positions=positions[:1], k_positions=positions)[:, 0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("normalize", [False, True])
def test_policy_disabled_and_invalid_terrain_preserve_nine_key_outputs(device, normalize, monkeypatch, record_property):
    obs = terrain_observations(device)
    kwargs = dict(actor_obs_normalization=normalize, critic_obs_normalization=normalize)
    torch.manual_seed(17)
    blind = make_policy(obs, **kwargs)
    torch.manual_seed(17)
    model = make_policy(obs, terrain_glimpse_cfg={}, **kwargs)
    # Constructing the new module must not alter seeded original weights.
    base_keys = set(blind.state_dict())
    assert {k for k in model.state_dict() if not k.startswith("terrain_glimpse.")} == base_keys
    assert blind.terrain_glimpse is None and blind.terrain_mode == "disabled"
    for key in base_keys:
        torch.testing.assert_close(model.state_dict()[key], blind.state_dict()[key], atol=0, rtol=0)
    assert sum(p.numel() for p in model.terrain_glimpse.parameters()) == 440968
    blind.update_normalization(obs)
    model.update_normalization(obs)
    with torch.no_grad():
        expected = legacy_encode(blind, obs)
        for actual, old in zip(blind._encode(obs), expected):
            torch.testing.assert_close(actual, old, atol=0, rtol=0)
        # Disabled reads neither map nor mask and never runs the module.
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError("disabled Terrain-Glimpse was evaluated")
            patch.setattr(model.terrain_glimpse, "forward", forbidden)
            for actual, old in zip(model._encode(obs.exclude("terrain_map", "terrain_valid")), expected):
                torch.testing.assert_close(actual, old, atol=0, rtol=0)
        model.terrain_mode = "enabled"
        obs["terrain_valid"].zero_()
        obs["terrain_map"].fill_(-3.)
        for actual, old in zip(model._encode(obs), expected):
            torch.testing.assert_close(actual, old, atol=1e-6, rtol=1e-5)
        record_property("invalid_terrain_intent_max_abs_error", (model._encode(obs)[1] - expected[1]).abs().max().item())
        torch.testing.assert_close(model.act_inference(obs), blind.act_inference(obs), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.evaluate(obs), blind.evaluate(obs), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [1, 3])
def test_policy_13_keys_use_existing_conditions_single_preprocessing_and_per_env_mask(device, n, record_property):
    obs = terrain_observations(device, n)
    if n == 3:
        obs["terrain_valid"][1, :, 10:] = False
        obs["terrain_valid"][2] = False
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled", actor_obs_normalization=True)
    # Calibrate on a separate batch, including for single-env inference. One
    # proprio sample has zero variance and artificially scales history by 100.
    model.update_normalization(terrain_observations(device, 32))
    # Fix four separated centers so mixed patch validity is deterministic.
    with torch.no_grad():
        centers = torch.tensor([[-.55, -.45], [.55, -.45], [-.55, .45], [.55, .45]], device=device)
        model.terrain_glimpse.selector.mlp[-1].weight.zero_()
        model.terrain_glimpse.selector.mlp[-1].bias.copy_(torch.atanh(centers.flatten() / .8))
    captured = {}
    def capture(name):
        def hook(module, args, kwargs, output):
            captured.setdefault(name, []).append((args, kwargs, output))
        return hook
    modules = dict(history=model.history_attention, commands=model.command_embed,
                   glimpse=model.terrain_glimpse, adapter=model.terrain_glimpse.input_adapter,
                   preprocess=model.terrain_glimpse.input_adapter.terrain_input, intent=model.intent_attention)
    handles = [module.register_forward_hook(capture(name), with_kwargs=True) for name, module in modules.items()]
    original = obs.clone()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    try:
        with torch.no_grad():
            _, intent = model._encode(obs)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(calls) == 1 for calls in captured.values())
    glimpse_args, _, terrain = captured["glimpse"][0]
    assert glimpse_args[0] is obs["terrain_map"] and glimpse_args[1] is obs["terrain_valid"]
    assert glimpse_args[2] is captured["history"][0][2]
    assert glimpse_args[3] is captured["commands"][0][0][0]
    torch.testing.assert_close(glimpse_args[3], model.command_obs_normalizer(obs["commands"].flatten(0, 1)).reshape(n, 8, 61))
    torch.testing.assert_close(captured["adapter"][0][2].terrain_map, obs["terrain_map"].clamp(-3, 3) / 3)
    (query, keys), positions, _ = captured["intent"][0]
    assert query is glimpse_args[2] and keys.shape == (n, 13, 128) and intent.shape == (n, 128)
    torch.testing.assert_close(keys[:, :1], query, atol=0, rtol=0)
    torch.testing.assert_close(keys[:, 1:9], captured["commands"][0][2], atol=0, rtol=0)
    torch.testing.assert_close(keys[:, 9:], terrain.terrain_tokens, atol=0, rtol=0)
    expected_mask = torch.tensor([[True]*4, [True, False, True, False], [False]*4], device=device)[:n]
    assert positions["q_positions"].tolist() == [0]
    assert positions["k_positions"].tolist() == [0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0]
    assert positions["key_valid"][:, :9].all()
    assert torch.equal(positions["key_valid"][:, 9:], expected_mask)
    expected, weights = numpy_attention(model.intent_attention, query, keys, positions["key_valid"], positions)
    np.testing.assert_allclose(intent.cpu(), expected[:, 0], atol=1e-6, rtol=1e-5)
    record_property("numpy_intent_max_abs_error", float(np.abs(intent.cpu().numpy() - expected[:, 0]).max()))
    assert not weights[..., 9:][~expected_mask.cpu().numpy()[:, None, None, :].repeat(4, 1)].any()
    for key in obs.keys():
        torch.testing.assert_close(obs[key], original[key], atol=0, rtol=0)
    for key in state:
        torch.testing.assert_close(model.state_dict()[key], state[key], atol=0, rtol=0)
    assert torch.equal(rng, torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["enabled", "disabled", "invalid"])
def test_policy_intent_gradient_reaches_crop_selector_only_with_valid_terrain(device, mode):
    obs = terrain_observations(device)
    obs["terrain_map"].requires_grad_()
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="disabled" if mode == "disabled" else "enabled")
    if mode == "invalid":
        obs["terrain_valid"].zero_()
    # Eliminate the encoder's direct center-coordinate path: selector gradient
    # must travel through the sampled heights. This is not yet a PPO test.
    with torch.no_grad():
        model.terrain_glimpse.patch_encoder.mlp[0].weight[:, -2:].zero_()
    _, intent = model._encode(obs)
    intent.square().sum().backward()
    for module in (model.terrain_glimpse.selector, model.terrain_glimpse.patch_encoder):
        grads = [p.grad for p in module.parameters()]
        if mode == "enabled":
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum().item() for g in grads) > 1e-8
        else:
            assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)
    map_grad = obs["terrain_map"].grad
    if mode == "enabled":
        assert map_grad is not None and torch.isfinite(map_grad).all() and map_grad.abs().sum() > 0
        changed = obs.detach().clone()
        changed["terrain_map"] += .3
        with torch.no_grad():
            assert (model._encode(changed)[1] - intent.detach()).abs().max() > 1e-6
    else:
        assert map_grad is None or torch.count_nonzero(map_grad) == 0


@pytest.mark.parametrize("kwargs", [
    {"terrain_mode": "auto"}, {"terrain_mode": "enabled"}, {"terrain_glimpse_cfg": False},
    {"terrain_glimpse_cfg": {}, "embedding_dim": 64}, {"terrain_glimpse_cfg": {"unknown": {}}},
    {"terrain_glimpse_cfg": {"crop_config": None}},
    {"terrain_glimpse_cfg": {"crop_config": {"min_valid_points": 26}}},
])
def test_policy_rejects_invalid_terrain_configuration(kwargs):
    with pytest.raises((ValueError, TypeError), match="PGMT"):
        make_policy(terrain_observations("cpu"), **kwargs)


def test_policy_mode_validation_and_configuration_metadata_round_trip():
    obs = terrain_observations("cpu")
    blind = make_policy(obs, embedding_dim=64)
    with pytest.raises(ValueError, match="constructed"):
        blind.terrain_mode = "enabled"
    assert blind.terrain_mode == "disabled"
    assert blind.get_checkpoint_metadata()["policy_family"] == "pgmt_stage1"
    assert "terrain_mode" not in blind.get_checkpoint_metadata()["policy_kwargs"]
    config = dict(selector_config={"hidden_dims": [64, 32], "center_limit_m": .7},
                  crop_config={"min_valid_points": 15}, encoder_config={"hidden_dims": [48]})
    model = make_policy(obs, terrain_glimpse_cfg=config, terrain_mode="enabled")
    config["selector_config"]["hidden_dims"][0] = 999
    metadata = model.get_checkpoint_metadata()
    assert metadata["policy_family"] == "pgmt_perception_only"
    assert metadata["policy_kwargs"]["terrain_glimpse_cfg"]["selector_config"]["hidden_dims"] == (64, 32)
    restored = PGMTActorCritic(obs, metadata["obs_groups"], 29, **metadata["policy_kwargs"])
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored._encode(obs)[1], model._encode(obs)[1], atol=0, rtol=0)
    model.terrain_mode = "disabled"
    assert model.get_checkpoint_metadata()["policy_kwargs"]["terrain_mode"] == "disabled"
    with pytest.raises(ValueError, match="terrain_mode"):
        model.terrain_mode = "typo"
    assert model.terrain_mode == "disabled"


@pytest.mark.parametrize("key", ["terrain_map", "terrain_valid"])
def test_policy_enabled_requires_canonical_map_pair(key):
    obs = terrain_observations("cpu")
    with pytest.raises(ValueError, match=key):
        make_policy(obs.exclude(key), terrain_glimpse_cfg={}, terrain_mode="enabled")
    model = make_policy(obs.exclude(key), terrain_glimpse_cfg={})
    model.terrain_mode = "enabled"
    with pytest.raises(ValueError, match=key):
        model._encode(obs.exclude(key))


@pytest.mark.parametrize("key,value", [
    ("terrain_map", torch.zeros(3, 20, 21)), ("terrain_valid", torch.zeros(3, 441)),
    ("terrain_map", torch.zeros(3, 21, 21, dtype=torch.float64)),
    ("terrain_valid", torch.zeros(3, 21, 21, dtype=torch.int64)),
])
def test_policy_reuses_glimpse_input_contract_validation(key, value):
    obs = terrain_observations("cpu")
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled")
    obs[key] = value
    with pytest.raises((ValueError, TypeError), match=key):
        model._encode(obs)


@pytest.mark.parametrize("device", DEVICES)
def test_actor_critic_share_saved_augmented_pair_and_recompute_without_cache(device):
    from pmt_tasks.pgmt_perception import terrain_observation_from_hits, pack_terrain_observation, split_terrain_observation
    obs = terrain_observations(device)
    hits = torch.zeros(3, 441, 3, device=device)
    sensor_z = torch.full((3,), 1., device=device)
    hits[..., 2] = sensor_z[:, None] - obs["terrain_map"].flatten(1)
    pair = terrain_observation_from_hits(hits, sensor_z, obs["terrain_valid"].flatten(1), terrain_observation_contract())
    obs.update(split_terrain_observation({"pgmt_terrain": pack_terrain_observation(pair)}))
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled",
                        actor_obs_normalization=True, critic_obs_normalization=True)
    model.update_normalization(terrain_observations(device, 32))
    calls, intents, preprocess = [], [], []
    handles = [
        model.terrain_glimpse.register_forward_hook(lambda m, a, o: calls.append((a, o))),
        model.intent_attention.register_forward_hook(lambda m, a, o: intents.append(o)),
        model.terrain_glimpse.input_adapter.terrain_input.register_forward_hook(lambda m, a, o: preprocess.append(o)),
    ]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    device_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    try:
        model._update_distribution(obs)
        action = model.act_inference(obs)
        value = model.evaluate(obs)
        assert action.shape == (3, 29) and value.shape == (3, 3)
        torch.testing.assert_close(action, model.action_mean, atol=0, rtol=0)
        assert len(calls) == len(intents) == len(preprocess) == 3
        for args, result in calls:
            assert args[0] is obs["terrain_map"] and args[1] is obs["terrain_valid"]
            assert result.debug is None and result.terrain_tokens.requires_grad
        for intent in intents:
            assert isinstance(intent, torch.Tensor) and intent.requires_grad
            torch.testing.assert_close(intent, intents[0], atol=0, rtol=0)
        assert intents[0] is not intents[1] and intents[1] is not intents[2]
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if device_rng is not None:
            assert torch.equal(device_rng, torch.cuda.get_rng_state())
        for key, original in state.items():
            torch.testing.assert_close(model.state_dict()[key], original, atol=0, rtol=0)
        # Reuse tensor addresses with changed saved observations: no stale tokens.
        obs["terrain_map"].add_(.25)
        assert (model.act_inference(obs) - action).abs().max() > 1e-7
        assert (model.evaluate(obs) - value).abs().max() > 1e-7
        assert len(calls) == 5
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("branch", ["actor", "critic"])
def test_actor_and_critic_losses_reach_same_glimpse_without_privileged_leak(device, branch, record_property):
    obs = terrain_observations(device)
    obs["critic"].requires_grad_()
    obs["terrain_map"].requires_grad_()
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled")
    with torch.no_grad():
        model.terrain_glimpse.patch_encoder.mlp[0].weight[:, -2:].zero_()
    output = model.act_inference(obs) if branch == "actor" else model.evaluate(obs)
    output.square().sum().backward()
    for name, module in (("selector", model.terrain_glimpse.selector),
                         ("patch_encoder", model.terrain_glimpse.patch_encoder), ("IFM", model.intent_attention)):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        norm = sum(g.abs().sum().item() for g in grads)
        assert norm > 1e-9
        record_property(f"{name}_gradient_l1", norm)
    assert obs["terrain_map"].grad.abs().sum() > 0
    if branch == "actor":
        assert obs["critic"].grad is None
    else:
        assert obs["critic"].grad.abs().sum() > 0
    changed = obs.detach().clone()
    changed["critic"].add_(2.)
    with torch.no_grad():
        torch.testing.assert_close(model.act_inference(changed), model.act_inference(obs), atol=0, rtol=0)
        torch.testing.assert_close(model._encode(changed)[1], model._encode(obs)[1], atol=0, rtol=0)
        assert (model.evaluate(changed) - model.evaluate(obs)).abs().max() > 1e-6


def diagnostic_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from diagnostic_tensors(item)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["enabled", "disabled", "invalid", "stage1"])
@pytest.mark.parametrize("training", [False, True])
def test_policy_debug_snapshots_have_no_state_rng_distribution_or_gradient_side_effects(device, mode, training):
    obs = terrain_observations(device)
    model = make_policy(obs, terrain_glimpse_cfg=None if mode == "stage1" else {},
                        terrain_mode="enabled" if mode in ("enabled", "invalid") else "disabled",
                        actor_obs_normalization=True, critic_obs_normalization=True)
    model.update_normalization(terrain_observations(device, 32))
    model.train(training)
    if mode == "invalid":
        obs["terrain_valid"].zero_()
    if mode in ("disabled", "stage1"):
        obs = obs.exclude("terrain_map", "terrain_valid")
    model._update_distribution(obs)
    distribution = model.distribution
    mean, std = model.action_mean.clone(), model.action_std.clone()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    original_obs = obs.clone()
    parameters = tuple(model.parameters())
    plain = model.act_inference(obs)
    plain_grad = torch.autograd.grad(plain.square().sum(), parameters, allow_unused=True)
    pending = model.act_inference(obs)
    cpu_rng = torch.get_rng_state()
    device_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    # Debug needs no privileged vector and must not update the Gaussian.
    debug = model.get_perception_debug(obs.exclude("critic"))
    assert isinstance(debug, PGMTPerceptionDebug)
    assert debug.terrain_mode == model.terrain_mode and model.training == training
    assert model.distribution is distribution
    torch.testing.assert_close(model.action_mean, mean, atol=0, rtol=0)
    torch.testing.assert_close(model.action_std, std, atol=0, rtol=0)
    torch.testing.assert_close(debug.intent, model._encode(obs)[1], atol=0, rtol=0)
    attention = debug.attention
    length = 13 if mode in ("enabled", "invalid") else 9
    assert attention.weights.shape == (3, 4, 1, length)
    assert len(debug.key_labels) == length and debug.key_labels[:2] == ("history", "command[0]")
    assert attention.key_valid[:, :9].all()
    assert attention.weights.masked_select(~attention.key_valid[:, None, None]).eq(0).all()
    torch.testing.assert_close(attention.weights.sum(-1), torch.ones(3, 4, 1, device=device))
    torch.testing.assert_close(debug.terrain_attention_mass, attention.weights[..., 9:].sum(-1), atol=0, rtol=0)
    if mode in ("enabled", "invalid"):
        assert debug.glimpse is not None
        assert debug.glimpse.center_xy_m.shape == (3, 4, 2)
        assert debug.glimpse.patch_height.shape == (3, 4, 5, 5)
        assert torch.equal(debug.glimpse.token_valid, attention.key_valid[:, 9:])
    else:
        assert debug.glimpse is None
    if mode != "enabled":
        assert debug.terrain_attention_mass.eq(0).all()
    else:
        assert debug.terrain_attention_mass.gt(0).all()
    originals = [t.clone() for t in diagnostic_tensors(debug)]
    tensors = list(diagnostic_tensors(debug))
    assert len({t.data_ptr() for t in tensors}) == len(tensors)
    for value in tensors:
        assert value.grad_fn is None and not value.requires_grad and value.device.type == device
        value.fill_(0) if value.dtype == torch.bool else value.add_(10)
    fresh = model.get_perception_debug(obs)
    for expected, actual in zip(originals, diagnostic_tensors(fresh)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    pending_grad = torch.autograd.grad(pending.square().sum(), parameters, allow_unused=True)
    for expected, actual in zip(plain_grad, pending_grad):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, original in state.items():
        torch.testing.assert_close(model.state_dict()[key], original, atol=0, rtol=0)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], original_obs[key], atol=0, rtol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if device_rng is not None:
        assert torch.equal(device_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("device", DEVICES)
def test_perception_obs_groups_route_renamed_fields_without_canonical_fallback(device):
    obs = terrain_observations(device)
    source = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled")
    metadata = source.get_checkpoint_metadata()
    groups = deepcopy(metadata["obs_groups"])
    groups.update(terrain_map=["saved_height"], terrain_valid=["saved_mask"])
    renamed = obs.clone()
    renamed["saved_height"], renamed["saved_mask"] = obs["terrain_map"], obs["terrain_valid"]
    # Decoys detect any hard-coded lookup of the old canonical keys.
    renamed["terrain_map"].fill_(-3.)
    renamed["terrain_valid"].zero_()
    model = PGMTActorCritic(renamed, groups, 29, **metadata["policy_kwargs"]).to(device)
    model.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model.act_inference(renamed), source.act_inference(obs), atol=0, rtol=0)
        torch.testing.assert_close(model.evaluate(renamed), source.evaluate(obs), atol=0, rtol=0)
    with pytest.raises(ValueError, match="saved_height"):
        model.evaluate(renamed.exclude("saved_height"))
    recorded = model.get_checkpoint_metadata()
    assert recorded["obs_schema"]["available_obs"]["saved_height"]["shape"] == [21, 21]
    assert recorded["signature"]["perception_contract"]["obs_groups"] == {
        "terrain_map": ["saved_height"], "terrain_valid": ["saved_mask"]}
    assert recorded["signature"] != metadata["signature"]


@pytest.mark.parametrize("mapping", [None, [], "terrain_map", ["terrain_map", "extra"], [1], [""]])
def test_perception_requires_explicit_single_key_observation_groups(mapping):
    obs = terrain_observations("cpu")
    groups = {**POLICY_GROUPS, "terrain_valid": ["terrain_valid"]}
    if mapping is not None:
        groups["terrain_map"] = mapping
    with pytest.raises(ValueError, match="obs_groups.*terrain_map"):
        PGMTActorCritic(obs, groups, 29, terrain_glimpse_cfg={})


@pytest.mark.parametrize("base_group", ["policy", "critic", "policy_history", "command_window", "terrain_valid"])
def test_perception_rejects_map_aliasing_mask_or_base_observations(base_group):
    groups = deepcopy({**POLICY_GROUPS, "terrain_map": ["terrain_map"], "terrain_valid": ["terrain_valid"]})
    groups[base_group].append("terrain_map") if base_group != "terrain_valid" else groups.update(terrain_valid=["terrain_map"])
    with pytest.raises(ValueError, match="PGMT terrain_map"):
        PGMTActorCritic(terrain_observations("cpu"), groups, 29, terrain_glimpse_cfg={})


@pytest.mark.parametrize("field,value", [
    ("terrain_map", torch.zeros(3, 441)), ("terrain_valid", torch.ones(3, 21, 21, dtype=torch.int64)),
    ("terrain_map", torch.zeros(3, 21, 21, dtype=torch.float64)),
    ("terrain_valid", torch.ones(3, 21, 21, device="meta")),
])
def test_enabled_policy_rejects_bad_map_contract_during_construction(field, value):
    obs = terrain_observations("cpu")
    obs[field] = value
    with pytest.raises((ValueError, TypeError), match=field):
        make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled")


@pytest.mark.parametrize("device", DEVICES)
def test_perceptive_metadata_records_shared_v3_contract_and_reconstructs_policy(device):
    from pmt_tasks.pgmt_perception import terrain_observation_contract as env_contract
    from pmt_tasks.pgmt_height_scan import SCAN_SHAPE
    assert env_contract is terrain_observation_contract
    obs = terrain_observations(device)
    declared = terrain_observation_contract(augmentation_enabled=False, nan_dropout_prob=.2, noise_range_m=(-.03, .03))
    model = make_policy(obs, terrain_glimpse_cfg={}, terrain_mode="enabled", terrain_observation_cfg=declared,
                        actor_obs_normalization=True, critic_obs_normalization=True)
    model.update_normalization(obs)
    metadata = model.get_checkpoint_metadata()
    contract = metadata["perception_contract"]
    assert contract["observation"] == declared and contract["observation"]["version"] == 3
    assert contract["observation"]["shape"] == list(SCAN_SHAPE)
    assert contract["fusion"]["key_rope_positions"] == [0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0]
    assert metadata["signature"]["value_groups"] == ["upper", "lower", "auxiliary"]
    restored = PGMTActorCritic(obs, metadata["obs_groups"], 29, **metadata["policy_kwargs"]).to(device)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.get_checkpoint_metadata() == metadata
    with torch.no_grad():
        torch.testing.assert_close(restored.act_inference(obs), model.act_inference(obs), atol=0, rtol=0)
        torch.testing.assert_close(restored.evaluate(obs), model.evaluate(obs), atol=0, rtol=0)
    # Caller/metadata mutation must not silently alter the live contract/groups.
    baseline = deepcopy(metadata)
    declared["noise_range_m"][0] = -9.
    metadata["obs_groups"]["terrain_map"][0] = "wrong"
    metadata["obs_schema"]["available_obs"]["terrain_map"]["shape"][0] = 99
    metadata["policy_kwargs"]["terrain_observation_cfg"]["units"] = "normalized"
    metadata["perception_contract"]["observation"]["version"] = 2
    assert model.get_checkpoint_metadata() == baseline
    model.terrain_mode = "disabled"
    assert model.get_checkpoint_metadata()["signature"] != baseline["signature"]
    assert model.get_checkpoint_metadata()["policy_family"] == "pgmt_perception_only"
    blind = make_policy(obs)
    old = blind.get_checkpoint_metadata()
    assert old["policy_family"] == "pgmt_stage1" and "perception_contract" not in old
    assert "terrain_observation_cfg" not in old["policy_kwargs"]
    assert "perception_contract" not in old["signature"]
    assert {k for k in model.state_dict() if not k.startswith("terrain_glimpse.")} == set(blind.state_dict()) | {
        f"{name}.{field}" for name in ("actor_obs_normalizer", "critic_obs_normalizer", "command_obs_normalizer")
        for field in ("_mean", "_var", "_std", "count")}


@pytest.mark.parametrize("key,value", [("version", 1), ("version", 2), ("units", "normalized"),
                                      ("height", "ground-sensor"), ("normalization_location", "observation")])
def test_policy_rejects_obsolete_perception_contract(key, value):
    declared = terrain_observation_contract()
    declared[key] = value
    with pytest.raises(ValueError, match="schema"):
        make_policy(terrain_observations("cpu"), terrain_glimpse_cfg={}, terrain_observation_cfg=declared)


def test_blind_policy_rejects_a_declared_perception_contract_without_module():
    with pytest.raises(ValueError, match="requires terrain_glimpse_cfg"):
        make_policy(terrain_observations("cpu"), terrain_observation_cfg=terrain_observation_contract())
