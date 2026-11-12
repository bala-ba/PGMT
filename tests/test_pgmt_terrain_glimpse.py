"""P2.2 standalone Terrain-Glimpse inputs, selection, crops and shared encoding."""
from dataclasses import asdict

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
from motion_tracking_rl.networks.pgmt_terrain_glimpse import (
    PGMTMaskedTerrainCrop,
    PGMTTerrainCropCfg,
    PGMTTerrainGlimpse,
    PGMTTerrainGlimpseInput,
    PGMTTerrainPatchEncoder,
    PGMTTerrainPatchEncoderCfg,
    PGMTTerrainPositionSelector,
    PGMTTerrainSelectorCfg,
)
from pmt_tasks.pgmt_perception import (
    pack_terrain_observation,
    split_terrain_observation,
    terrain_observation_contract,
    terrain_observation_from_hits,
)


def inputs(n=2, device="cpu", dtype=torch.float32):
    return dict(
        terrain_map=torch.linspace(-4, 4, n * 441, device=device, dtype=dtype).reshape(n, 21, 21),
        terrain_valid=torch.ones(n, 21, 21, device=device, dtype=torch.bool),
        history_token=torch.ones(n, 1, 128, device=device, dtype=dtype),
        commands=torch.full((n, 8, 61), 7.0, device=device, dtype=dtype),
    )


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_metric_normalization_preserves_pmt_sign_invalid_values_and_axis_order(n, mask_dtype):
    data = inputs(n)
    # Non-contiguous map exercises the row=y, col=x contract without reshaping.
    data["terrain_map"] = data["terrain_map"].transpose(1, 2)
    data["terrain_map"][:, 0, :6] = torch.tensor([-4., -3., -.5, 0., 1.5, 4.])
    data["terrain_valid"] = data["terrain_valid"].to(mask_dtype)
    data["terrain_valid"][:, 0, 1:3] = 0
    before = {key: value.clone() for key, value in data.items()}

    result = PGMTTerrainGlimpseInput()(**data)
    expected = torch.tensor([-1., -1., -1. / 6., 0., .5, 1.]).expand(n, -1)
    torch.testing.assert_close(result.terrain_map[:, 0, :6], expected, rtol=0, atol=0)
    torch.testing.assert_close(result.terrain_map, data["terrain_map"].clamp(-3, 3) / 3, rtol=0, atol=0)
    torch.testing.assert_close(result.terrain_valid, data["terrain_valid"].float(), rtol=0, atol=0)
    # Already-normalized state/commands are neither normalized again nor detached.
    assert result.history_token is data["history_token"]
    assert result.commands is data["commands"]
    for key, value in data.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [.1, 1.])
def test_atomic_augmented_observation_is_reusable_without_new_rng_or_mutation(dropout):
    torch.manual_seed(2201)
    hits = torch.zeros(4, 441, 3)
    sensor_z = torch.full((4,), .8)
    map_valid = torch.ones(4, 441, dtype=torch.bool)
    map_valid[:, :20] = False
    pair = terrain_observation_from_hits(hits, sensor_z, map_valid,
                                        terrain_observation_contract(nan_dropout_prob=dropout))
    obs = split_terrain_observation({"pgmt_terrain": pack_terrain_observation(pair)})
    data = inputs(4)
    data.update(obs)
    snapshots = {key: value.clone() for key, value in data.items()}
    rng = torch.random.get_rng_state().clone()
    adapter = PGMTTerrainGlimpseInput()
    first = adapter(**data)
    second = adapter(**data)
    adapter.eval()
    third = adapter(**data)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not adapter.state_dict()  # No normalizer statistics or new policy weights.
    for a, b, c in zip(first, second, third):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
        assert torch.isfinite(a).all()
    torch.testing.assert_close(first.terrain_map, pair["height"] / 3, rtol=0, atol=0)
    torch.testing.assert_close(first.terrain_valid, pair["valid"].float(), rtol=0, atol=0)
    assert first.terrain_map[~pair["valid"]].eq(-1).all()
    for key, value in data.items():
        torch.testing.assert_close(value, snapshots[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_input_adapter_keeps_autograd_and_device(device):
    data = inputs(device=device)
    data["terrain_map"] = torch.full((2, 21, 21), .9, device=device)
    for key in ("terrain_map", "history_token", "commands"):
        data[key].requires_grad_()
    rng = (torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()).clone()
    prepared = PGMTTerrainGlimpseInput().to(device)(**data)
    loss = prepared.terrain_map.sum() + 2 * prepared.history_token.sum() + 3 * prepared.commands.sum()
    loss.backward()
    for key, expected in (("terrain_map", 1. / 3.), ("history_token", 2.), ("commands", 3.)):
        gradient = data[key].grad
        torch.testing.assert_close(gradient, torch.full_like(gradient, expected))
        assert torch.isfinite(gradient).all()
    assert all(value.device == data["terrain_map"].device for value in prepared)
    after = torch.cuda.get_rng_state() if device == "cuda" else torch.random.get_rng_state()
    assert torch.equal(rng, after)


@pytest.mark.parametrize("name,shape", [
    ("terrain_map", (2, 441)), ("terrain_map", (2, 20, 21)),
    ("terrain_valid", (2, 21, 20)), ("history_token", (2, 128)),
    ("history_token", (2, 1, 64)), ("commands", (2, 7, 61)),
    ("commands", (2, 8, 60)),
    ("terrain_valid", (1, 21, 21)), ("history_token", (1, 1, 128)),
    ("commands", (1, 8, 61)),
])
def test_rejects_misshapen_or_mismatched_batches(name, shape):
    data = inputs()
    data[name] = torch.ones(shape)
    with pytest.raises(ValueError, match=name):
        PGMTTerrainGlimpseInput()(**data)


@pytest.mark.parametrize("name,dtype", [
    ("terrain_map", torch.int64), ("terrain_valid", torch.int64),
    ("history_token", torch.int64), ("commands", torch.int64),
    ("history_token", torch.float64), ("commands", torch.float64),
])
def test_rejects_implicit_feature_dtype_conversion(name, dtype):
    data = inputs()
    data[name] = data[name].to(dtype)
    with pytest.raises(TypeError, match=name):
        PGMTTerrainGlimpseInput()(**data)


@pytest.mark.parametrize("name", ["terrain_valid", "history_token", "commands"])
def test_rejects_implicit_device_transfer(name):
    data = inputs()
    data[name] = data[name].to("meta")
    with pytest.raises(ValueError, match=f"{name} device"):
        PGMTTerrainGlimpseInput()(**data)


def test_accepts_actual_stage1_history_and_normalized_commands_without_changing_policy():
    n = 2
    obs = TensorDict({"proprio": torch.randn(n, 96), "history": torch.randn(n, 10, 96),
                      "commands": torch.randn(n, 8, 61) + 5., "critic": torch.randn(n, 251)}, [n])
    groups = {"policy": ["proprio"], "policy_history": ["history"],
              "command_window": ["commands"], "critic": ["critic"]}
    policy = PGMTActorCritic(obs, groups, 29, actor_hidden_dims=[16], critic_hidden_dims=[16],
                            actor_obs_normalization=True, critic_obs_normalization=True)
    policy.update_normalization(obs)
    policy.eval()
    captured = {}

    def capture_history(module, args, output):
        captured["history_token"] = output

    def capture_commands(module, args, output):
        captured["commands"] = output.reshape(n, 8, 61)

    hooks = [policy.history_attention.register_forward_hook(capture_history),
             policy.command_obs_normalizer.register_forward_hook(capture_commands)]
    try:
        expected_action = policy.act_inference(obs)
    finally:
        for hook in hooks:
            hook.remove()
    original_state = {key: value.clone() for key, value in policy.state_dict().items()}
    data = inputs(n)
    data.update(captured)
    result = PGMTTerrainGlimpseInput()(**data)
    assert result.history_token is captured["history_token"]
    assert result.commands is captured["commands"]
    assert not torch.equal(result.commands, obs["commands"])
    torch.testing.assert_close(policy.act_inference(obs), expected_action, rtol=0, atol=0)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 4])
def test_selector_uses_prepared_height_mask_state_and_commands_in_order(n):
    data = inputs(n)
    data["terrain_map"] = torch.full((n, 21, 21), .9).transpose(1, 2)
    data["terrain_map"][:, 3, 5] = 1.5
    data["terrain_valid"][:, 7, 8] = False
    data["history_token"][:, 0, 9] = 4.
    data["commands"][:, 6, 13] = 8.
    adapter = PGMTTerrainGlimpseInput()
    selector = PGMTTerrainPositionSelector()
    calls, features = [], []
    pre_hook = adapter.terrain_input.register_forward_hook(lambda module, args, out: calls.append(1))
    mlp_hook = selector.mlp.register_forward_pre_hook(lambda module, args: features.append(args[0].detach().clone()))
    try:
        prepared = adapter(**data)
        centers = selector(prepared)
        selector(prepared)
    finally:
        pre_hook.remove()
        mlp_hook.remove()
    assert calls == [1]
    assert centers.shape == (n, 4, 2)
    packed = features[0]
    assert packed.shape == (n, 1498)
    assert packed[:, 3 * 21 + 5].eq(.5).all()
    torch.testing.assert_close(packed[:, 0], torch.full((n,), .9) / 3, atol=0, rtol=0)
    assert packed[:, 441 + 7 * 21 + 8].eq(0).all()
    assert packed[:, 441 + 7 * 21 + 9].eq(1).all()
    assert packed[:, 882 + 9].eq(4).all()
    assert packed[:, 1010 + 6 * 61 + 13].eq(8).all()
    torch.testing.assert_close(features[0], features[1], atol=0, rtol=0)
    assert data["terrain_map"][0, 3, 5] == 1.5


@pytest.mark.parametrize("limit", [.8, .6])
def test_selector_xy_pairing_and_saturation_keep_future_crop_inside_scan(limit):
    selector = PGMTTerrainPositionSelector(PGMTTerrainSelectorCfg(center_limit_m=limit))
    raw = torch.tensor([100., -100., .25, -.5, -.75, 1., -1., .5])
    with torch.no_grad():
        selector.mlp[-1].weight.zero_()
        selector.mlp[-1].bias.copy_(raw)
    prepared = PGMTTerrainGlimpseInput()(**inputs())
    centers = selector(prepared)
    torch.testing.assert_close(centers, (limit * raw.tanh()).reshape(1, 4, 2).expand(2, -1, -1))
    assert centers.abs().max() <= limit
    assert (centers - .2).min() >= -1.
    assert (centers + .2).max() <= 1.


def test_selector_repeated_forward_and_batch_permutation_preserve_saved_observation():
    torch.manual_seed(2212)
    data = inputs(4)
    # Include the finite sentinel/mask pair produced by a wholly invalid scan.
    data["terrain_map"][0].fill_(-3.)
    data["terrain_valid"][0].fill_(False)
    prepared = PGMTTerrainGlimpseInput()(**data)
    snapshots = [value.clone() for value in prepared]
    selector = PGMTTerrainPositionSelector()
    rng = torch.random.get_rng_state().clone()
    first = selector(prepared)
    selector.eval()
    second = selector(prepared)
    order = torch.tensor([2, 0, 3, 1])
    reordered = selector(type(prepared)(*(value[order] for value in prepared)))
    assert torch.isfinite(first).all()
    assert first.abs().max() <= .8
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(reordered, first[order], rtol=1e-6, atol=1e-7)
    assert torch.equal(rng, torch.random.get_rng_state())
    for value, snapshot in zip(prepared, snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["terrain_map", "terrain_valid", "history_token", "commands"])
def test_selector_centers_respond_to_each_condition(field):
    torch.manual_seed(2213)
    data = inputs()
    selector = PGMTTerrainPositionSelector()
    adapter = PGMTTerrainGlimpseInput()
    baseline = selector(adapter(**data))
    if field == "terrain_valid":
        data[field].logical_not_()
    else:
        data[field] = data[field] + .3
    changed = selector(adapter(**data))
    assert (changed - baseline).abs().max() > 1e-6


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_selector_gradients_reach_all_layers_and_condition_inputs(device):
    torch.manual_seed(2214)
    data = inputs(device=device)
    # Non-flat, unclipped map: gradient must reach the metric observation.
    data["terrain_map"] = torch.linspace(-1., 1., 882, device=device).reshape(2, 21, 21)
    for field in ("terrain_map", "history_token", "commands"):
        data[field].requires_grad_()
    selector = PGMTTerrainPositionSelector().to(device)
    centers = selector(PGMTTerrainGlimpseInput().to(device)(**data))
    loss = (centers - torch.tensor([.2, -.1], device=device)).square().mean()
    loss.backward()
    for parameter in selector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for field in ("terrain_map", "history_token", "commands"):
        assert torch.isfinite(data[field].grad).all()
        assert data[field].grad.abs().sum() > 0
    assert centers.device == data["terrain_map"].device


def test_selector_center_derivatives_match_finite_difference():
    torch.manual_seed(2215)
    data = inputs(1, dtype=torch.float64)
    data["terrain_map"] = torch.linspace(-1., 1., 441, dtype=torch.float64).reshape(1, 21, 21)
    selector = PGMTTerrainPositionSelector().double()
    adapter = PGMTTerrainGlimpseInput()
    for field in ("terrain_map", "history_token", "commands"):
        data[field].requires_grad_()
    selector(adapter(**data))[0, 0, 0].backward()
    epsilon = 1e-5
    for field, index in (("terrain_map", (0, 3, 4)), ("history_token", (0, 0, 0)),
                         ("commands", (0, 7, 60))):
        with torch.no_grad():
            plus = {key: value.clone() for key, value in data.items()}
            minus = {key: value.clone() for key, value in data.items()}
            plus[field][index] += epsilon
            minus[field][index] -= epsilon
            numerical = (selector(adapter(**plus))[0, 0, 0] - selector(adapter(**minus))[0, 0, 0]) / (2 * epsilon)
        torch.testing.assert_close(data[field].grad[index], numerical, atol=1e-9, rtol=1e-5)


def test_selector_config_and_weights_round_trip():
    config = PGMTTerrainSelectorCfg(hidden_dims=(32, 16), center_limit_m=.7)
    selector = PGMTTerrainPositionSelector(config)
    restored = PGMTTerrainPositionSelector(PGMTTerrainSelectorCfg(**asdict(config)))
    restored.load_state_dict(selector.state_dict(), strict=True)
    prepared = PGMTTerrainGlimpseInput()(**inputs())
    torch.testing.assert_close(restored(prepared), selector(prepared), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"hidden_dims": ()}, {"hidden_dims": (0,)}, {"hidden_dims": (-1,)},
    {"hidden_dims": (True,)}, {"hidden_dims": (32.,)},
    {"center_limit_m": 0.}, {"center_limit_m": .81}, {"center_limit_m": float("nan")},
    {"center_limit_m": float("inf")}, {"center_limit_m": True},
])
def test_selector_rejects_config_that_breaks_network_or_crop_bounds(kwargs):
    with pytest.raises(ValueError, match="PGMT selector"):
        PGMTTerrainSelectorCfg(**kwargs)


def test_selector_rejects_raw_observation_dictionary():
    with pytest.raises(TypeError, match="PGMTTerrainGlimpseInputs"):
        PGMTTerrainPositionSelector()(inputs())


def prepared_map(height_m, valid=None):
    """Use the real network preprocessing exactly once for analytic samples."""
    n = height_m.shape[0]
    return PGMTTerrainGlimpseInput()(
        height_m, torch.ones_like(height_m, dtype=torch.bool) if valid is None else valid,
        height_m.new_zeros(n, 1, 128), height_m.new_zeros(n, 8, 61))


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("slope", [(0., 0.), (.6, 0.), (0., -.9), (.6, -.9)])
def test_crop_matches_analytic_planes_at_fractional_centers(n, slope):
    axis = torch.linspace(-1, 1, 21, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ax, ay = slope
    height_m = (.7 + ax * xx + ay * yy).expand(n, -1, -1)
    centers = torch.tensor([[.023, .037], [-.157, .213], [.8, -.8], [-.8, .8]],
                           dtype=torch.float64).expand(n, -1, -1)
    result = PGMTMaskedTerrainCrop()(prepared_map(height_m), centers)
    delta = torch.arange(-2, 3, dtype=torch.float64) / 10
    expected = (.7 + ax * (centers[..., 0, None, None] + delta[None, :])
                + ay * (centers[..., 1, None, None] + delta[:, None])) / 3
    torch.testing.assert_close(result.height, expected, atol=1e-14, rtol=1e-13)
    assert result.point_valid.all() and result.patch_valid.all()
    torch.testing.assert_close(result.valid_weight, torch.ones_like(result.valid_weight))
    assert result.coverage.eq(1).all()


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_crop_axis_order_for_yaw_aligned_world_slope(quarter_turns):
    # World plane: ground_z = .4*world_x - .2*world_y. Supply its sensor-ground
    # map in four actual torso-yaw frames; crop coordinates stay in that frame.
    cosine, sine = [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)][quarter_turns]
    axis = torch.linspace(-1, 1, 21, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    ground = .4 * (cosine * x - sine * y) - .2 * (sine * x + cosine * y)
    centers = torch.tensor([[[.033, -.027], [.213, .157], [-.4, .4], [.4, -.4]]], dtype=torch.float64)
    result = PGMTMaskedTerrainCrop()(prepared_map((.8 - ground)[None]), centers)
    dy, dx = torch.meshgrid(torch.arange(-2, 3, dtype=torch.float64) / 10,
                           torch.arange(-2, 3, dtype=torch.float64) / 10, indexing="ij")
    px = centers[..., 0, None, None] + dx
    py = centers[..., 1, None, None] + dy
    expected = (.8 - .4 * (cosine * px - sine * py) + .2 * (sine * px + cosine * py)) / 3
    torch.testing.assert_close(result.height, expected, atol=1e-14, rtol=1e-13)


def test_crop_stair_edge_has_expected_bilinear_height():
    height = torch.full((1, 21, 21), .3, dtype=torch.float64)
    height[:, :, 10:] = .9  # Discontinuity between x=-0.1 and x=0.0 grid points.
    centers = torch.tensor([[[-.05, 0.]] * 4], dtype=torch.float64)
    result = PGMTMaskedTerrainCrop()(prepared_map(height), centers)
    expected_row = torch.tensor([.1, .1, .2, .3, .3], dtype=torch.float64)
    torch.testing.assert_close(result.height, expected_row.expand(1, 4, 5, 5), atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("invalid_fill", [-3., 3., float("nan"), float("inf")])
def test_crop_renormalizes_support_and_ignores_invalid_source_heights(invalid_fill):
    height = torch.full((1, 21, 21), .6, dtype=torch.float64)
    valid = torch.ones_like(height, dtype=torch.bool)
    height[0, 10, 11] = invalid_fill
    valid[0, 10, 11] = False
    centers = torch.tensor([[[.025, 0.], [.05, 0.], [.075, 0.], [.1, 0.]]], dtype=torch.float64)
    result = PGMTMaskedTerrainCrop()(prepared_map(height, valid), centers)
    torch.testing.assert_close(result.valid_weight[0, :, 2, 2],
                               torch.tensor([.75, .5, .25, 0.], dtype=torch.float64), atol=1e-14, rtol=0)
    assert result.point_valid[0, :, 2, 2].tolist() == [True, True, False, False]
    torch.testing.assert_close(result.height[0, :, 2, 2],
                               torch.tensor([.2, .2, -1., -1.], dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.isfinite(result.height).all()


def test_crop_treats_nonfinite_height_as_missing_even_if_input_mask_is_true():
    height = torch.full((1, 21, 21), .6, dtype=torch.float64)
    height[0, 10, 10] = torch.nan
    result = PGMTMaskedTerrainCrop()(prepared_map(height), torch.zeros(1, 4, 2, dtype=torch.float64))
    assert not result.point_valid[0, :, 2, 2].any()
    assert result.height[0, :, 2, 2].eq(-1).all()
    assert torch.isfinite(result.height).all()


def test_crop_patch_validity_uses_count_of_points_not_average_interpolation_weight():
    height = torch.full((2, 21, 21), .6)
    valid = torch.zeros_like(height, dtype=torch.bool)
    for env, count in enumerate([12, 13]):
        patch = torch.arange(25).reshape(5, 5) < count
        valid[env, 8:13, 8:13] = patch
    prepared = prepared_map(height, valid)
    centers = torch.zeros(2, 4, 2)
    result = PGMTMaskedTerrainCrop()(prepared, centers)
    assert result.point_valid.sum((-2, -1)).tolist() == [[12] * 4, [13] * 4]
    assert result.patch_valid.tolist() == [[False] * 4, [True] * 4]
    torch.testing.assert_close(result.coverage, torch.tensor([[.48] * 4, [.52] * 4]))
    strict = PGMTMaskedTerrainCrop(PGMTTerrainCropCfg(min_valid_points=25))(prepared, centers)
    assert not strict.patch_valid.any()


@pytest.mark.parametrize("dimension,sign", [(0, 1.), (0, -1.), (1, 1.), (1, -1.)])
def test_crop_outside_scan_is_invalid_even_when_padding_has_partial_valid_support(dimension, sign):
    height = torch.full((1, 21, 21), .6, dtype=torch.float64)
    centers = torch.zeros(1, 4, 2, dtype=torch.float64)
    centers[0, :, dimension] = sign * torch.tensor([.8, 1., 1.025, 2.], dtype=torch.float64)
    result = PGMTMaskedTerrainCrop()(prepared_map(height), centers)
    # x/y=1.025 would retain 75% border weight without explicit in-scan gating.
    assert result.point_valid.sum((-2, -1)).tolist() == [[25, 15, 10, 0]]
    assert result.patch_valid.tolist() == [[True, True, False, False]]
    assert result.height[~result.point_valid].eq(-1).all()
    assert result.valid_weight[0, 3].eq(0).all()
    torch.testing.assert_close(result.height[result.point_valid],
                               torch.full_like(result.height[result.point_valid], .2))


def test_crop_all_invalid_and_invalid_coordinates_have_finite_zero_gradients():
    height = torch.full((1, 21, 21), -3., dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[[.023, .037], [torch.nan, 0.], [torch.inf, 0.], [1e200, 0.]]],
                           dtype=torch.float64, requires_grad=True)
    result = PGMTMaskedTerrainCrop()(prepared_map(height, torch.zeros_like(height, dtype=torch.bool)), centers)
    assert result.height.eq(-1).all()
    assert result.valid_weight.eq(0).all() and result.coverage.eq(0).all()
    assert not result.point_valid.any() and not result.patch_valid.any()
    result.height.sum().backward()
    assert height.grad.eq(0).all() and centers.grad.eq(0).all()


def test_masked_crop_center_derivative_matches_finite_difference():
    axis = torch.linspace(-1, 1, 21, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    height = (.7 + .6 * x - .4 * y)[None].requires_grad_()
    valid = torch.ones_like(height, dtype=torch.bool)
    valid[0, 11, 11] = False
    prepared = prepared_map(height, valid)
    centers = torch.tensor([[[.023, .037], [-.153, .213], [.333, -.127], [-.457, -.223]]],
                           dtype=torch.float64, requires_grad=True)
    crop = PGMTMaskedTerrainCrop()
    result = crop(prepared, centers)
    assert result.point_valid[:, :, 2, 2].all()
    assert 0.5 < result.valid_weight[0, 0, 2, 2] < 1.0  # Exercise derivative of denominator.
    result.height[:, :, 2, 2].sum().backward()
    assert height.grad[0, 11, 11] == 0
    assert height.grad.abs().sum() > 0
    for glimpse in range(4):
        for dim in range(2):
            plus, minus = centers.detach().clone(), centers.detach().clone()
            plus[0, glimpse, dim] += 1e-6
            minus[0, glimpse, dim] -= 1e-6
            with torch.no_grad():
                derivative = (crop(prepared, plus).height[:, :, 2, 2].sum()
                              - crop(prepared, minus).height[:, :, 2, 2].sum()) / 2e-6
            torch.testing.assert_close(centers.grad[0, glimpse, dim], derivative, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_crop_height_alone_backpropagates_through_selector(device):
    torch.manual_seed(2231)
    axis = torch.linspace(-1, 1, 21, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    prepared = prepared_map((.7 + .6 * x - .4 * y)[None])
    selector = PGMTTerrainPositionSelector().to(device)
    crop = PGMTMaskedTerrainCrop().to(device)
    centers = selector(prepared)
    centers.retain_grad()
    patches = crop(prepared, centers)
    patches.height.mean().backward()  # No center concatenation or auxiliary selector loss.
    assert torch.isfinite(centers.grad).all() and centers.grad.abs().sum() > 0
    for parameter in selector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("fractional", [False, True])
def test_crop_ten_percent_dropout_retains_patches_without_height_bias(fractional):
    torch.manual_seed(2232)
    valid = torch.rand(1024, 21, 21) >= .1
    height = torch.where(valid, .6, -3.)
    centers = torch.tensor([[[-.6, -.6], [-.6, .6], [.6, -.6], [.6, .6]]]).expand(1024, -1, -1)
    if fractional:
        centers = centers + torch.tensor([.033, .021])
    prepared = prepared_map(height, valid)
    snapshots = [value.clone() for value in prepared]
    crop = PGMTMaskedTerrainCrop()
    rng = torch.random.get_rng_state().clone()
    result = crop(prepared, centers)
    repeated = crop(prepared, centers)
    assert result.patch_valid.float().mean() > .99
    assert result.point_valid.float().mean() > .88
    torch.testing.assert_close(result.height[result.point_valid],
                               torch.full_like(result.height[result.point_valid], .2), atol=1e-7, rtol=1e-6)
    for first, second in zip(result, repeated):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    for value, snapshot in zip(prepared, snapshots):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("kwargs", [
    {"min_valid_weight": 0.}, {"min_valid_weight": 1.1}, {"min_valid_weight": float("nan")},
    {"min_valid_points": 0}, {"min_valid_points": 26}, {"min_valid_points": True},
    {"weight_epsilon": 0.}, {"weight_epsilon": .6}, {"weight_epsilon": float("nan")},
])
def test_crop_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError, match="PGMT crop"):
        PGMTTerrainCropCfg(**kwargs)


@pytest.mark.parametrize("shape", [(1, 4, 2), (2, 3, 2), (2, 4, 3)])
def test_crop_rejects_wrong_center_shape_or_batch(shape):
    prepared = PGMTTerrainGlimpseInput()(**inputs(2))
    with pytest.raises(ValueError, match="center_xy_m"):
        PGMTMaskedTerrainCrop()(prepared, torch.zeros(shape))


def test_patch_encoder_feature_order_weight_sharing_and_patch_permutation():
    torch.manual_seed(2241)
    data = inputs(2)
    data["terrain_valid"][:, ::3, ::2] = False
    centers = torch.tensor([[[-.623, -.317], [.243, -.137], [-.127, .413], [.513, .327]]]).expand(2, -1, -1)
    patches = PGMTMaskedTerrainCrop()(PGMTTerrainGlimpseInput()(**data), centers)
    encoder = PGMTTerrainPatchEncoder()
    seen = []
    hook = encoder.mlp.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    result = encoder(patches, centers)
    hook.remove()
    assert result.shape == (2, 4, 128)
    assert len(seen) == 1 and seen[0].shape == (2, 4, 52)
    torch.testing.assert_close(seen[0][..., :25], patches.height.flatten(-2), rtol=0, atol=0)
    torch.testing.assert_close(seen[0][..., 25:50], patches.point_valid.float().flatten(-2), rtol=0, atol=0)
    torch.testing.assert_close(seen[0][..., 50:], centers, rtol=0, atol=0)
    # One parameter set, independent of the four patch indices or batch size.
    assert sum(p.numel() for p in encoder.parameters()) == 23_296
    permutation = torch.tensor([3, 0, 2, 1])
    reordered = type(patches)(*(value[:, permutation] for value in patches))
    torch.testing.assert_close(encoder(reordered, centers[:, permutation]), result[:, permutation])
    identical = type(patches)(*(value[:, :1].expand_as(value) for value in patches))
    repeated = encoder(identical, centers[:, :1].expand_as(centers))
    torch.testing.assert_close(repeated, repeated[:, :1].expand_as(repeated), rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 4])
def test_glimpse_debug_is_independent_and_preprocessing_runs_once(n):
    torch.manual_seed(2242)
    model = PGMTTerrainGlimpse()
    data = inputs(n)
    for key in ("terrain_map", "history_token", "commands"):
        data[key].requires_grad_()
    before = {key: value.detach().clone() for key, value in data.items()}
    prepared, captured = [], []
    input_hook = model.input_adapter.terrain_input.register_forward_hook(
        lambda _, args, output: prepared.append(output))
    encoder_hook = model.patch_encoder.register_forward_pre_hook(lambda _, args: captured.append(args))
    rng = torch.random.get_rng_state().clone()
    plain = model(**data)
    inspected = model(**data, return_debug=True)
    input_hook.remove()
    encoder_hook.remove()
    assert len(prepared) == 2  # Exactly once for each complete forward.
    torch.testing.assert_close(prepared[1][0], data["terrain_map"].clamp(-3, 3) / 3, rtol=0, atol=0)
    assert plain.debug is None and inspected.debug is not None
    assert plain.terrain_tokens.shape == (n, 4, 128) and plain.token_valid.shape == (n, 4)
    assert plain.terrain_tokens.requires_grad and inspected.terrain_tokens.requires_grad
    torch.testing.assert_close(plain.terrain_tokens, inspected.terrain_tokens, rtol=0, atol=0)
    patches, centers = captured[1]
    main_values = (centers, patches.height, patches.point_valid, patches.valid_weight,
                   patches.coverage, inspected.token_valid)
    for snapshot, main in zip(inspected.debug, main_values):
        assert not snapshot.requires_grad and snapshot.grad_fn is None
        assert snapshot.device == main.device and snapshot.data_ptr() != main.data_ptr()
        torch.testing.assert_close(snapshot, main, rtol=0, atol=0)
        snapshot.fill_(0)  # Display-side edits cannot modify saved autograd tensors.
    torch.testing.assert_close(inspected.token_valid, plain.token_valid, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    first_grads = torch.autograd.grad(plain.terrain_tokens.square().mean(), parameters)
    second_grads = torch.autograd.grad(inspected.terrain_tokens.square().mean(), parameters)
    for first, second in zip(first_grads, second_grads):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        evaluated = model(**data)
    torch.testing.assert_close(evaluated.terrain_tokens, plain.terrain_tokens, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in data.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_glimpse_propagates_partial_and_all_invalid_masks_without_reselecting():
    model = PGMTTerrainGlimpse()
    # Force identical centers at an aligned position for exact 12/25,13/25 cases.
    with torch.no_grad():
        model.selector.mlp[-1].weight.zero_()
        model.selector.mlp[-1].bias.zero_()
    data = inputs(3)
    data["terrain_map"].fill_(-3.)
    data["terrain_valid"].zero_()
    for env, count in enumerate([0, 12, 13]):
        data["terrain_valid"][env, 8:13, 8:13] = torch.arange(25).reshape(5, 5) < count
    data["terrain_map"][data["terrain_valid"]] = .6
    result = model(**data, return_debug=True)
    assert torch.isfinite(result.terrain_tokens).all()
    assert result.token_valid.dtype == torch.bool
    assert result.token_valid.tolist() == [[False] * 4, [False] * 4, [True] * 4]
    debug = result.debug
    assert debug.center_xy_m.eq(0).all()
    assert debug.point_valid.sum((-2, -1)).tolist() == [[0] * 4, [12] * 4, [13] * 4]
    torch.testing.assert_close(debug.coverage, torch.tensor([[0.] * 4, [.48] * 4, [.52] * 4]))
    assert debug.patch_height[~debug.point_valid].eq(-1).all()
    torch.testing.assert_close(debug.token_valid, result.token_valid)
    # A future loss that excludes invalid tokens contributes no encoder/selector gradient.
    excluded_loss = (result.terrain_tokens[:2] * result.token_valid[:2, :, None]).sum()
    excluded_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and parameter.grad.eq(0).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_glimpse_token_loss_reaches_selector_through_crop_without_center_feature_shortcut(device):
    torch.manual_seed(2243)
    model = PGMTTerrainGlimpse().to(device)
    # Remove the direct center-feature contribution while leaving crop geometry live.
    with torch.no_grad():
        model.patch_encoder.mlp[0].weight[:, 50:].zero_()
    axis = torch.linspace(-1, 1, 21, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    data = inputs(2, device=device)
    data["terrain_map"] = (.7 + .6 * x - .4 * y)[None].repeat(2, 1, 1)
    for key in ("terrain_map", "history_token", "commands"):
        data[key].requires_grad_()
    result = model(**data, return_debug=True)
    assert result.token_valid.all() and torch.isfinite(result.terrain_tokens).all()
    result.terrain_tokens.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    for key in ("terrain_map", "history_token", "commands"):
        assert torch.isfinite(data[key].grad).all() and data[key].grad.abs().sum() > 0


def test_glimpse_configuration_and_weights_roundtrip(tmp_path):
    model = PGMTTerrainGlimpse(PGMTTerrainSelectorCfg(hidden_dims=(64,), center_limit_m=.7),
                               PGMTTerrainCropCfg(min_valid_points=20),
                               PGMTTerrainPatchEncoderCfg(hidden_dims=[96, 64], activation="relu"))
    configs = dict(selector_config=asdict(model.selector.config), crop_config=asdict(model.crop.config),
                   encoder_config=asdict(model.patch_encoder.config))
    path = tmp_path / "standalone_glimpse.pt"
    torch.save(dict(configs=configs, state_dict=model.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = PGMTTerrainGlimpse(PGMTTerrainSelectorCfg(**saved["configs"]["selector_config"]),
                                  PGMTTerrainCropCfg(**saved["configs"]["crop_config"]),
                                  PGMTTerrainPatchEncoderCfg(**saved["configs"]["encoder_config"]))
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert all("debug" not in name for name in saved["state_dict"])
    data = inputs(1)
    first, second = model(**data, return_debug=True), restored(**data, return_debug=True)
    for a, b in zip(first[:2] + first.debug, second[:2] + second.debug):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"hidden_dims": ()}, {"hidden_dims": (0,)}, {"hidden_dims": (True,)},
    {"hidden_dims": (1.5,)}, {"activation": None},
])
def test_patch_encoder_rejects_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError), match="PGMT patch encoder"):
        PGMTTerrainPatchEncoderCfg(**kwargs)


@pytest.mark.parametrize("problem", ["raw_tuple", "height_shape", "mask_dtype", "center_shape", "center_dtype"])
def test_patch_encoder_rejects_incompatible_inputs(problem):
    centers = torch.zeros(2, 4, 2)
    patches = PGMTMaskedTerrainCrop()(PGMTTerrainGlimpseInput()(**inputs(2)), centers)
    if problem == "raw_tuple":
        patches = tuple(patches)
    elif problem == "height_shape":
        patches = patches._replace(height=patches.height[..., :4])
    elif problem == "mask_dtype":
        patches = patches._replace(point_valid=patches.point_valid.float())
    elif problem == "center_shape":
        centers = centers[:, :3]
    else:
        centers = centers.double()
    with pytest.raises((ValueError, TypeError), match="PGMT patch encoder"):
        PGMTTerrainPatchEncoder()(patches, centers)
