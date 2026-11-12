"""Exact PMT sign/fill/sampling parity and atomic PGMT observation transport."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.networks.pgmt_terrain_input import PGMTTerrainInput
from motion_tracking_rl.storage.rollout_storage import RolloutStorage
from pmt_tasks.pgmt_perception import (terrain_observation_contract, validate_terrain_observation_contract,
    terrain_observation_from_hits, pack_terrain_observation, split_terrain_observation)
from pmt_tasks.pgmt_runtime import PGMTState

ROOT = Path(__file__).resolve().parents[1]


def inputs(n=4):
    z = torch.linspace(-2., 2., n)
    relative = torch.linspace(-2., 2., 441).expand(n, -1)
    hits = torch.zeros(n, 441, 3)
    hits[..., 2] = z[:, None] - relative
    valid = torch.ones(n, 441, dtype=torch.bool)
    valid[:, :42] = False
    return hits, z, valid


def original_observation_function():
    path = ROOT / 'pmt_tasks/mdp/observations.py'
    names = {'_apply_hit_z_dropout', '_apply_uniform_noise', '_apply_clip', 'height_scan_for_vision'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    ns = dict(torch=torch, ManagerBasedEnv=object, SceneEntityCfg=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['height_scan_for_vision']


@pytest.mark.parametrize('enabled', [True, False])
def test_exact_original_pmt_parity_real_world_coordinates_and_consecutive_calls(enabled):
    hits, z, valid = inputs()
    hits[:, 42:45, 2] = torch.tensor([float('nan'), float('inf'), -float('inf')])
    hits[:, 45:50, 2] = z[:, None] - torch.tensor([-4., -3., 0., 3., 4.])
    z[0] = float('nan')  # PMT's sensor-height fallback, independent of hit mask.
    # Original PMT has no map boundary mask: present out-of-map cells as missing rays.
    pmt_hits = hits.clone()
    pmt_hits[..., 2].masked_fill_(~valid, float('nan'))
    pos = torch.zeros(4, 3)
    pos[:, 2] = z
    sensor = SimpleNamespace(data=SimpleNamespace(ray_hits_w=pmt_hits, pos_w=pos))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={'scan': sensor}))
    original = original_observation_function()
    torch.manual_seed(4529)
    expected = [original(env, SimpleNamespace(name='scan'), nan_dropout_prob=.1 if enabled else 0.,
                         append_validity_mask=True, noise_range=(-.06, .06) if enabled else None,
                         clip_range=(-3., 3.)) for _ in range(3)]
    torch.manual_seed(4529)
    actual = [pack_terrain_observation(terrain_observation_from_hits(hits, z, valid,
              terrain_observation_contract(augmentation_enabled=enabled))) for _ in range(3)]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    assert torch.equal(actual[0], actual[1]) is (not enabled)


def test_world_z_fill_is_30_not_constant_relative_height_or_zero():
    hits, z, valid = inputs(1)
    z[:] = 29.5
    hits[:, :, 2] = float('nan')
    result = terrain_observation_from_hits(hits, z, valid, terrain_observation_contract(augmentation_enabled=False))
    assert result['height'].eq(-.5).all()  # 29.5 - 30; no observation-side /3.
    assert not result['valid'].any()
    z[:] = .8
    result = terrain_observation_from_hits(hits, z, valid, terrain_observation_contract(augmentation_enabled=False))
    assert result['height'].eq(-3.).all()


def test_sign_clipping_and_network_normalization_location():
    hits, z, valid = inputs(1)
    z[:] = .8
    valid.fill_(True)
    hits[..., 2] = 0.
    hits[:, :5, 2] = z[:, None] - torch.tensor([-4., -3., 0., 3., 4.])
    before = torch.random.get_rng_state().clone()
    result = terrain_observation_from_hits(hits, z, valid, terrain_observation_contract(augmentation_enabled=False))
    torch.testing.assert_close(result['height'].flatten()[:5], torch.tensor([-3., -3., 0., 3., 3.]))
    assert result['height'].flatten()[5] == .8  # PMT sign: floor below torso is positive.
    normalized, mask = PGMTTerrainInput()(result['height'], result['valid'])
    torch.testing.assert_close(normalized, result['height'] / 3, atol=0, rtol=0)
    assert mask.eq(1).all()
    assert torch.equal(before, torch.random.get_rng_state())


def test_network_preprocessor_matches_real_pmt_network_method():
    path = ROOT / 'motion_tracking_rl/networks/vision_transformer_actor_critic.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'VisionTransformerActorCritic')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_map_encoding')
    ns = dict(torch=torch, TensorDict=TensorDict)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    pair = terrain_observation_from_hits(*inputs(), terrain_observation_contract())
    fake = SimpleNamespace(map_height=21, map_width=21, z_clip=3., normalize_height=True,
                           _grid_x=torch.zeros(21, 21), _grid_y=torch.zeros(21, 21),
                           _split_height_scan=lambda scan, count: scan.split(count, -1),
                           map_transformer=lambda data, *args: data)
    pmt = ns['_get_map_encoding'](fake, {'vision': pack_terrain_observation(pair)}, None, None)
    actual, valid = PGMTTerrainInput()(pair['height'], pair['valid'])
    torch.testing.assert_close(actual, pmt[:, 2], atol=0, rtol=0)
    torch.testing.assert_close(valid, pmt[:, 3], atol=0, rtol=0)
    assert actual[~pair['valid']].eq(-1).all()


def test_statistics_and_raw_sensor_isolation():
    torch.manual_seed(6201)
    hits, z, valid = inputs(256)
    before = (hits.clone(), z.clone(), valid.clone())
    pair = terrain_observation_from_hits(hits, z, valid, terrain_observation_contract())
    kept = pair['valid'].flatten(1)
    residual = (pair['height'].flatten(1) - (z[:, None] - hits[..., 2]))[kept]
    rate = float((valid & ~kept).sum() / valid.sum())
    assert .095 < rate < .105
    assert abs(float(residual.mean())) < .0004
    assert .034 < float(residual.std()) < .0353
    assert residual.min() >= -.060001 and residual.max() <= .060001
    assert not kept[~valid].any()
    assert pair['height'][~pair['valid']].eq(-3).all()
    for original, snapshot in zip((hits, z, valid), before):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize('missing_geometry', [True, False])
def test_all_invalid_or_all_dropped_are_finite_pmt_sentinels(missing_geometry):
    hits, z, valid = inputs()
    if missing_geometry:
        valid.zero_()
    pair = terrain_observation_from_hits(hits, z, valid,
             terrain_observation_contract(nan_dropout_prob=.1 if missing_geometry else 1.))
    assert not pair['valid'].any()
    assert pair['height'].eq(-3).all()


def test_runtime_resamples_same_step_and_all_envs_after_partial_reset():
    hits, z, valid = inputs()
    pos = torch.zeros(4, 3)
    pos[:, 2] = z
    state = PGMTState.__new__(PGMTState)
    state.env = SimpleNamespace(common_step_counter=0, cfg=SimpleNamespace(pgmt_terrain_observation=terrain_observation_contract()),
        scene=SimpleNamespace(sensors={'height_scanner': SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits, pos_w=pos))},
                              terrain=SimpleNamespace(contains=lambda xy: valid)))
    state._obs_version = 0
    state.pending_reset = torch.zeros(4, dtype=torch.bool)
    state.previous_target = torch.zeros(4, 29)
    state.qtarget = lambda: torch.ones(4, 29)
    first = state.terrain_observations()
    snapshot = {k: v.clone() for k, v in first.items()}
    second = state.terrain_observations()
    assert ((first['height'] != second['height']).flatten(1).any(1)).all()
    state.reset(torch.tensor([2]))
    third = state.terrain_observations()
    for key in first:
        torch.testing.assert_close(first[key], snapshot[key], atol=0, rtol=0)
    assert ((second['height'] != third['height']).flatten(1).any(1)).all()
    assert ((second['valid'] != third['valid']).flatten(1).any(1)).all()
    assert state.pending_reset.tolist() == [False, False, True, False]


def test_atomic_pair_split_and_native_wrapper_preserve_rng_and_other_observations():
    pair = terrain_observation_from_hits(*inputs(), terrain_observation_contract())
    raw = dict(pgmt_terrain=pack_terrain_observation(pair), proprio=torch.zeros(4, 96))
    before = torch.random.get_rng_state().clone()
    result = split_terrain_observation(raw)
    assert 'pgmt_terrain' in raw and 'pgmt_terrain' not in result
    torch.testing.assert_close(result['terrain_map'], pair['height'], atol=0, rtol=0)
    torch.testing.assert_close(result['terrain_valid'], pair['valid'].float(), atol=0, rtol=0)
    path = ROOT / 'pmt_tasks/isaaclab_rl/rsl_rl/vecenv_wrapper.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_observation_tensordict')
    ns = dict(TensorDict=TensorDict)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    wrapped = ns['_observation_tensordict'](SimpleNamespace(num_envs=4), raw)
    for key in result:
        torch.testing.assert_close(wrapped[key], result[key], atol=0, rtol=0)
    legacy = {'vision': torch.ones(4, 374)}
    assert split_terrain_observation(legacy) is legacy
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('kwargs', [dict(nan_dropout_prob=p) for p in (-.1, 1.1, float('nan'), float('inf'))]
    + [dict(noise_range_m=r) for r in ((.1,-.1), (0.,float('nan')), (0.,float('inf')), (0.,))]
    + [dict(augmentation_enabled='false')])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        terrain_observation_contract(**kwargs)


@pytest.mark.parametrize('key,value', [('version', 1), ('version', 2), ('height', 'ground-torso'),
    ('units', 'dimensionless'), ('invalid_world_z_fill_m', 0.), ('normalization_location', 'observation'),
    ('noise_refresh', 'control_step_or_selected_env_reset'), ('shape', [11,11])])
def test_schema_change_fails(key, value):
    cfg = terrain_observation_contract()
    validate_terrain_observation_contract(cfg)
    cfg[key] = value
    with pytest.raises(ValueError, match='schema'):
        validate_terrain_observation_contract(cfg)


def test_rollout_minibatches_preserve_samples_and_network_does_not_resample():
    n, steps = 4, 3
    source, storage = {}, None
    for step in range(steps):
        pair = terrain_observation_from_hits(*inputs(n), terrain_observation_contract())
        obs = TensorDict(split_terrain_observation(dict(pgmt_terrain=pack_terrain_observation(pair),
              sample_id=torch.arange(step*n,(step+1)*n)[:,None].float())), [n])
        if storage is None:
            storage = RolloutStorage('rl', n, steps, obs, [29], num_value_heads=3)
        t = storage.Transition()
        t.observations, t.actions = obs, torch.zeros(n,29)
        t.rewards = t.values = torch.zeros(n,3)
        t.dones = t.actions_log_prob = torch.zeros(n)
        t.action_mean, t.action_sigma = t.actions, torch.ones_like(t.actions)
        storage.add_transitions(t)
        for e in range(n):
            source[step*n+e] = (obs['terrain_map'][e].clone(), obs['terrain_valid'][e].clone())
    for epoch in range(2):
        seen = []
        for batch, *_ in storage.mini_batch_generator(3,1):
            rng = torch.random.get_rng_state().clone()
            norm, mask = PGMTTerrainInput()(batch['terrain_map'], batch['terrain_valid'])
            assert torch.equal(rng, torch.random.get_rng_state())
            for i, sample_id in enumerate(batch['sample_id'].flatten().long().tolist()):
                seen.append(sample_id)
                height, valid = source[sample_id]
                torch.testing.assert_close(batch['terrain_map'][i], height, atol=0, rtol=0)
                torch.testing.assert_close(batch['terrain_valid'][i], valid, atol=0, rtol=0)
                torch.testing.assert_close(norm[i], height/3, atol=0, rtol=0)
                torch.testing.assert_close(mask[i], valid, atol=0, rtol=0)
        assert sorted(seen) == list(range(n*steps))
