"""P2.3 full-policy CPU/CUDA fusion acceptance on saved version-3 maps.

Run from PMT with the pmt Python environment. Requires CUDA and never starts
Isaac. Reports strict old-weight equivalence, real-map/debug/masking checks,
the existing regression suite, and full-width policy/native PPO costs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import gc
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_tracking_rl.algorithms import PPO
from motion_tracking_rl.networks import PGMTActorCritic
from scripts.validate_pgmt_terrain_glimpse import load_maps, map_cases, numpy_crop, sha256
from scripts.validate_pgmt_perception_ppo import FILES as REGRESSIONS

GROUPS = {"policy": ["proprio"], "critic": ["critic"],
          "policy_history": ["history"], "command_window": ["commands"]}
TERRAIN_GROUPS = {**GROUPS, "terrain_map": ["terrain_map"], "terrain_valid": ["terrain_valid"]}
ATOL = RTOL = 1e-5


def observations(height, valid, seed):
    """Generate base observations on CPU once, then copy them across devices."""
    generator = torch.Generator().manual_seed(seed)
    n = len(height)
    return TensorDict(dict(proprio=torch.randn(n, 96, generator=generator),
                           critic=torch.randn(n, 251, generator=generator),
                           history=torch.randn(n, 10, 96, generator=generator),
                           commands=torch.randn(n, 8, 61, generator=generator),
                           terrain_map=height.clone(), terrain_valid=valid.float()), [n])


def make_policy(obs, *, terrain, device="cpu"):
    # Full Stage-1 widths, including empirical proprio/command/critic statistics.
    with redirect_stdout(io.StringIO()):
        policy = PGMTActorCritic(
            obs, TERRAIN_GROUPS if terrain else GROUPS, 29,
            actor_obs_normalization=True, critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            terrain_glimpse_cfg={} if terrain else None,
            terrain_mode="enabled" if terrain else "disabled")
    return policy.to(device)


def copy_old_state(blind, policy):
    """Test fixture only: explicit checked copy, not the P2.4 migration API."""
    old, new = blind.state_dict(), policy.state_dict()
    additions = set(new) - set(old)
    assert additions and all(key.startswith("terrain_glimpse.") for key in additions)
    assert set(old) == {key for key in new if not key.startswith("terrain_glimpse.")}
    for key, value in old.items():
        assert value.shape == new[key].shape and value.dtype == new[key].dtype, key
    merged = {key: old[key] if key in old else value for key, value in new.items()}
    policy.load_state_dict(merged, strict=True)
    for key, value in old.items():
        torch.testing.assert_close(policy.state_dict()[key], value, atol=0, rtol=0)
    normalizers = [key for key in old if "normalizer" in key]
    assert len(normalizers) == 12 and all(int(old[key]) > 0 for key in normalizers if key.endswith("count"))
    return dict(copied_keys=sorted(old), new_keys=sorted(additions), normalizer_keys=normalizers,
                strict_load=True, every_copied_tensor_exact=True)


def old_outputs(model, obs):
    """Explicit nine-key encoding from before perception integration."""
    proprio = model.actor_obs_normalizer(obs["proprio"])
    history = model.actor_obs_normalizer(obs["history"].flatten(0, 1)).reshape(-1, 10, 96)
    commands = model.command_obs_normalizer(obs["commands"].flatten(0, 1)).reshape(-1, 8, 61)
    token = model.history_attention(model.proprio_embed(proprio[:, None]), model.proprio_embed(history))
    keys = torch.cat((token, model.command_embed(commands)), 1)
    positions = torch.tensor([0, 0, 1, 3, 7, 15, 31, 63, 127], device=keys.device)
    intent = model.intent_attention(token, keys, q_positions=positions[:1], k_positions=positions)[:, 0]
    return dict(intent=intent, action_mean=model.actor(torch.cat((proprio, intent), -1)),
                value=model.critic(torch.cat((model.critic_obs_normalizer(obs["critic"]), intent), -1)))


def outputs(policy, obs):
    return dict(intent=policy._encode(obs)[1], action_mean=policy.act_inference(obs), value=policy.evaluate(obs))


def compare(first, second, *, atol=ATOL, rtol=RTOL):
    assert first.keys() == second.keys()
    errors = {}
    for key in first:
        a, b = first[key], second[key]
        if a.dtype == torch.bool:
            assert torch.equal(a, b), key
            errors[key] = 0.
        else:
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), key
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda msg: f"{key}: {msg}")
            errors[key] = float((a - b).abs().max())
    return errors


def detached(values):
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def real_map_audit(blind_state, terrain_state, obs_cpu, labels, device, saved_count):
    obs = obs_cpu.to(device)
    blind = make_policy(obs_cpu[:1], terrain=False, device=device)
    policy = make_policy(obs_cpu[:1], terrain=True, device=device)
    blind.load_state_dict(blind_state, strict=True)
    policy.load_state_dict(terrain_state, strict=True)
    for key, value in blind.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[key], value, atol=0, rtol=0)
    buffers = {name: value.clone() for name, value in policy.named_buffers()}
    inputs = obs.clone()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    report = {}
    with torch.no_grad():
        legacy = old_outputs(blind, obs)
        report["stage1_vs_explicit_old_path"] = compare(outputs(blind, obs), legacy)
        policy.terrain_mode = "disabled"
        disabled = outputs(policy, obs)
        report["disabled_vs_old"] = compare(disabled, legacy)
        policy.terrain_mode = "enabled"
        missing = obs.clone()
        missing["terrain_valid"].zero_()
        missing["terrain_map"].fill_(-3.)
        all_invalid = outputs(policy, missing)
        report["all_invalid_vs_old"] = compare(all_invalid, legacy)
        active = outputs(policy, obs)
        proprio, intent, debug = policy._encode(obs, return_debug=True)
        from_debug = dict(intent=intent, action_mean=policy.actor(torch.cat((proprio, intent), -1)),
                          value=policy.critic(torch.cat((policy.critic_obs_normalizer(obs["critic"]), intent), -1)))
        report["normal_vs_debug"] = compare(active, from_debug, atol=0, rtol=0)
        report["public_debug_intent"] = compare({"intent": policy.get_perception_debug(obs).intent},
                                                {"intent": active["intent"]}, atol=0, rtol=0)
        mask = debug.attention.key_valid
        weights = debug.attention.weights
        assert mask.shape == (len(obs), 13) and weights.shape == (len(obs), 4, 1, 13)
        assert mask[:, :9].all() and torch.equal(mask[:, 9:], debug.glimpse.token_valid)
        assert debug.attention.k_positions.tolist() == [0, 0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0]
        assert torch.isfinite(weights).all() and weights[~mask[:, None, None, :].expand_as(weights)].eq(0).all()
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6, rtol=1e-6)
        assert not debug.terrain_attention_mass[-saved_count:].any()  # Last case: all invalid.
        # Perturb already encoded invalid K/V, not raw invalid map heights (the
        # selector reads those heights and may move other, still-valid patches).
        def replace_masked_tokens(module, args, kwargs):
            query, keys = args
            valid = kwargs["key_valid"]
            altered = torch.where(valid[..., None], keys, torch.full_like(keys, 71.))
            return (query, altered), kwargs

        handle = policy.intent_attention.register_forward_pre_hook(replace_masked_tokens, with_kwargs=True)
        try:
            report["masked_token_perturbation"] = compare(outputs(policy, obs), active, atol=0, rtol=0)
        finally:
            handle.remove()
        arrays = detached(dict(**active, center_xy_m=debug.glimpse.center_xy_m,
                               patch_height=debug.glimpse.patch_height, point_valid=debug.glimpse.point_valid,
                               valid_weight=debug.glimpse.valid_weight, patch_valid=debug.glimpse.token_valid,
                               key_valid=mask, attention_weights=weights, terrain_attention_mass=debug.terrain_attention_mass))
        oracle = numpy_crop(obs_cpu["terrain_map"].numpy(), obs_cpu["terrain_valid"].bool().numpy(),
                            arrays["center_xy_m"].numpy())
        for actual, expected in zip((arrays["patch_height"], arrays["valid_weight"],
                                     arrays["point_valid"], arrays["patch_valid"]), oracle):
            if actual.dtype == torch.bool:
                assert np.array_equal(actual.numpy(), expected)
            else:
                np.testing.assert_allclose(actual.numpy(), expected, atol=ATOL, rtol=RTOL)
        report["numpy_crop"] = dict(points=int(arrays["patch_height"].numel()), masks_exact=True,
            max_height_error=float(np.max(np.abs(arrays["patch_height"].numpy() - oracle[0]))))
        single_errors = {key: 0. for key in active}
        for index in range(len(obs)):
            one = obs[index:index + 1]
            single = outputs(policy, one)
            errors = compare(single, {key: value[index:index + 1] for key, value in active.items()})
            for key, value in errors.items():
                single_errors[key] = max(single_errors[key], value)
            single_debug = policy.get_perception_debug(one)
            torch.testing.assert_close(single_debug.attention.weights, weights[index:index + 1], atol=ATOL, rtol=RTOL)
            assert torch.equal(single_debug.attention.key_valid, mask[index:index + 1])
        report["single_vs_mixed_batch"] = dict(samples=len(obs), max_abs_errors=single_errors)
        # All 20 source maps, including augmented L9, retain their exact mask.
        changed = obs[:saved_count].clone()
        changed["terrain_map"] = torch.where(changed["terrain_valid"] > .5,
                                              (changed["terrain_map"] + .3).clamp(-3., 3.), changed["terrain_map"])
        modified = outputs(policy, changed)
        response = {key: (modified[key] - active[key][:saved_count]).abs().amax(-1) for key in active}
        assert mask[:saved_count, 9:].any(-1).all(), "Sensitivity needs at least one valid patch per source map"
        assert all((value > 1e-8).all() for value in response.values()), response
        report["valid_map_response"] = dict(per_map=[dict(label=labels[i], **{k: float(v[i]) for k, v in response.items()})
                                                    for i in range(saved_count)],
                                            minimum_max_abs_change={k: float(v.min()) for k, v in response.items()})
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for key in obs.keys():
        torch.testing.assert_close(obs[key], inputs[key], atol=0, rtol=0)
    for name, value in policy.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    report.update(normalizer_buffers_unchanged=True, rng_unchanged=True, inputs_unchanged=True,
                  per_map=[dict(label=label, token_valid=arrays["patch_valid"][i].tolist(),
                      terrain_attention_per_head=arrays["terrain_attention_mass"][i, :, 0].tolist())
                      for i, label in enumerate(labels)])
    return report, dict(**arrays, **{f"old_{k}": v for k, v in detached(legacy).items()},
                       **{f"disabled_{k}": v for k, v in detached(disabled).items()},
                       **{f"invalid_{k}": v for k, v in detached(all_invalid).items()})


def measure(function, *, device, warmup, iterations, prepare=lambda: None, after=lambda: None):
    """One synchronized API call per sample; setup and assertions are untimed."""
    is_cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        prepare()
        function()
        after()
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elapsed, cuda_elapsed, memory = [], [], []
    for _ in range(iterations):
        prepare()
        if is_cuda:
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        function()
        if is_cuda:
            end_event.record()
            torch.cuda.synchronize(device)
        elapsed.append(1000 * (time.perf_counter() - start))
        if is_cuda:
            cuda_elapsed.append(start_event.elapsed_time(end_event))
            peak = torch.cuda.max_memory_allocated(device)
            memory.append(dict(baseline_allocated_MiB=baseline / 2**20, peak_allocated_MiB=peak / 2**20,
                               extra_peak_MiB=(peak - baseline) / 2**20,
                               peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 2**20))
        after()
    report = dict(warmup=warmup, samples=iterations, wall_ms=elapsed,
                  median_ms=float(np.median(elapsed)), p95_ms=float(np.quantile(elapsed, .95)))
    if is_cuda:
        report.update(cuda_event_ms=cuda_elapsed, cuda_event_median_ms=float(np.median(cuda_elapsed)),
                      cuda_event_p95_ms=float(np.quantile(cuda_elapsed, .95)),
                      **{key: max(row[key] for row in memory) for key in memory[0]})
    return report


def fill_rollout(algorithm, obs, generator):
    """Native rollout/GAE on saved scans; timed update does not include this."""
    n = len(obs)
    device = obs["proprio"].device
    with torch.inference_mode():
        for step in range(algorithm.storage.num_transitions_per_env):
            algorithm.act(obs)
            rewards = .2 * torch.randn(n, 3, generator=generator).to(device)
            dones = torch.zeros(n, dtype=torch.bool, device=device)
            if step == 2:
                dones[::7] = True
            algorithm.process_env_step(obs, rewards.sum(-1), dones, {"reward_groups": rewards})
        algorithm.compute_returns(obs)
    algorithm.optimizer.zero_grad(set_to_none=True)


def grad_l1(module):
    total = 0.
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            total += float(parameter.grad.abs().sum())
        assert torch.isfinite(parameter).all()
    return total


def benchmark(state, source_obs, args, device):
    records = []
    for n in args.batches:
        index = torch.arange(n) % len(source_obs)
        obs_cpu = source_obs[index]
        obs = obs_cpu.to(device)
        for mode in ("disabled", "enabled"):
            policy = make_policy(obs_cpu[:1], terrain=True, device=device)
            policy.load_state_dict(state, strict=True)
            policy.terrain_mode = mode
            for api in ("act_inference", "evaluate", "actor_plus_critic"):
                def forward():
                    with torch.inference_mode():
                        if api in ("act_inference", "actor_plus_critic"):
                            policy.act_inference(obs)
                        if api in ("evaluate", "actor_plus_critic"):
                            policy.evaluate(obs)

                result = measure(forward, device=device, warmup=args.warmup, iterations=args.iterations)
                records.append(dict(device=str(device), batch=n, terrain_mode=mode, api=api, **result))
            del policy
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            policy = make_policy(obs_cpu[:1], terrain=True, device=device)
            policy.load_state_dict(state, strict=True)
            policy.terrain_mode = mode
            algorithm = PPO(policy, device=str(device), num_learning_epochs=2, num_mini_batches=4,
                            learning_rate=1e-4, schedule="fixed", debug_raise_on_nonfinite=True)
            algorithm.init_storage("rl", n, 4, obs, [29])
            generator = torch.Generator().manual_seed(args.seed + n)
            metrics, gradients = [], []

            def update():
                metrics.append(algorithm.update())

            def check_update():
                assert algorithm.storage.step == 0
                assert all(np.isfinite(value) for value in metrics[-1].values())
                grads = dict(selector=grad_l1(policy.terrain_glimpse.selector),
                             encoder=grad_l1(policy.terrain_glimpse.patch_encoder),
                             ifm=grad_l1(policy.intent_attention))
                assert grads["ifm"] > 0
                if mode == "enabled":
                    assert grads["selector"] > 0 and grads["encoder"] > 0
                else:
                    assert grads["selector"] == grads["encoder"] == 0
                gradients.append(grads)

            result = measure(update, device=device, warmup=1, iterations=args.ppo_iterations,
                             prepare=lambda: fill_rollout(algorithm, obs, generator), after=check_update)
            steps = [int(state["step"]) for state in algorithm.optimizer.state.values()]
            assert steps and min(steps) == max(steps) == 8 * (1 + args.ppo_iterations)
            records.append(dict(device=str(device), batch=n, terrain_mode=mode, api="ppo_update",
                num_envs=n, rollout_steps=4, minibatch_size=n, minibatches=4, epochs=2,
                optimizer_steps_per_update=8, optimizer_steps_including_warmup=max(steps),
                metrics=metrics, last_minibatch_gradients=gradients, **result))
            print(f"Benchmark passed: device={device}, batch={n}, mode={mode}, PPO median={result['median_ms']:.3f} ms", flush=True)
            del algorithm, policy, metrics, gradients
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        del obs, obs_cpu
    return records


def save_plot(obs, labels, arrays, path):
    """Offline scientific QA figure: physical heights, glimpses and IFM mass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(4, 5, figsize=(15, 11), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        height = obs["terrain_map"][index].numpy()
        mask = obs["terrain_valid"][index].bool().numpy()
        ax.imshow(np.ma.masked_where(~mask, height), origin="lower", extent=(-1.05, 1.05, -1.05, 1.05),
                  cmap="terrain", vmin=-.5, vmax=2.)
        ax.set_facecolor("#444444")
        for patch, (x, y) in enumerate(arrays["center_xy_m"][index].numpy()):
            color = ["#e63946", "#00aaff", "#c000ff", "#ff8c00"][patch]
            ax.add_patch(Rectangle((x - .2, y - .2), .4, .4, fill=False, edgecolor=color, lw=1.5))
            ax.text(x, y, str(patch), color=color, fontsize=9)
        level, family = labels[index].split(":")
        mass = arrays["terrain_attention_mass"][index, :, 0].numpy()
        ax.set_title(f"{family} | {level.replace('terrain_map_', '').replace('pmt_augmented_maps_', 'aug_').replace('.npz', '')}\n"
                     + "terrain attention: " + "/".join(f"{value:.2f}" for value in mass), fontsize=8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle("Untrained full-policy fixture: saved metric sensor-ground maps and 4 glimpses\n"
                 "Gray = invalid; attention shows each of 4 heads; colors identify patch indices", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args, report, output):
    if not torch.cuda.is_available():
        raise RuntimeError("Visible CUDA is required; GPU acceptance must not silently skip")
    device = torch.device(args.device)
    assert device.type == "cuda"
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["environment"] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        cuda_build=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device),
        cpu_threads=torch.get_num_threads(), matmul_precision=torch.get_float32_matmul_precision(), tf32=False,
        nvidia_smi_before=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True))
    height, valid, labels, report["inputs"] = load_maps(args.source)
    # Retain the previous comparison fixtures and all native PPO regressions.
    command = [sys.executable, "-m", "pytest", "-q", "-o", "junit_family=legacy", *REGRESSIONS,
               f"--junitxml={output / 'pytest.xml'}"]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "pytest.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    suites = ET.parse(output / "pytest.xml").getroot()
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    assert counts["skipped"] == 0, counts
    report["pytest"] = dict(command=command, **counts)
    print(f"Related regression passed: {counts}", flush=True)
    source_obs = observations(height, valid, args.seed)
    torch.manual_seed(args.seed)
    blind = make_policy(source_obs, terrain=False)
    calibration = observations(height.repeat(4, 1, 1), valid.repeat(4, 1, 1), args.seed + 1)
    blind.update_normalization(calibration)
    # Deliberately construct different base weights, then explicitly copy them.
    torch.manual_seed(args.seed + 2)
    policy = make_policy(source_obs, terrain=True)
    assert not torch.equal(policy.actor[0].weight, blind.actor[0].weight)
    report["weight_fixture"] = copy_old_state(blind, policy)
    blind_state = {key: value.clone() for key, value in blind.state_dict().items()}
    state = {key: value.clone() for key, value in policy.state_dict().items()}
    report["policy"] = dict(actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        actor_input=224, critic_input=379, actions=29, values=3, embedding=128, heads=4,
        parameters=sum(p.numel() for p in policy.parameters()),
        terrain_parameters=sum(p.numel() for p in policy.terrain_glimpse.parameters()),
        metadata=policy.get_checkpoint_metadata(), dtype="float32", fixture_seed=args.seed,
        normalizer_counts={name: int(module.count) for name, module in policy.named_modules() if hasattr(module, "count")})
    del policy, blind, calibration
    cases = map_cases(height, valid, args.seed)
    case_obs = observations(torch.cat([value[0] for value in cases.values()]),
                            torch.cat([value[1] for value in cases.values()]), args.seed + 3)
    case_labels = [f"{label}/{name}" for name in cases for label in labels]
    report["coverage"] = dict(source_maps=len(height), cases=list(cases), samples_per_device=len(case_obs),
                              single_sample_checks_per_device=len(case_obs), mixed_batch_size=len(case_obs))
    cpu, cpu_arrays = real_map_audit(blind_state, state, case_obs, case_labels, "cpu", len(height))
    print("Real maps: CPU fusion, strict fallback, debug, masking and N=1 checks passed", flush=True)
    cuda, cuda_arrays = real_map_audit(blind_state, state, case_obs, case_labels, device, len(height))
    report["functional"] = dict(cpu=cpu, cuda=cuda, cpu_cuda_max_abs_errors=compare(cpu_arrays, cuda_arrays),
                                atol=ATOL, rtol=RTOL)
    np.savez_compressed(output / "fusion_debug.npz", labels=np.array(case_labels),
        terrain_map_m=case_obs["terrain_map"].numpy(), terrain_valid=case_obs["terrain_valid"].numpy(),
        **{key: value.numpy() for key, value in cuda_arrays.items()})
    save_plot(case_obs, labels, cuda_arrays, output / "fusion_real_maps.png")
    print("Real maps: CUDA and CPU/CUDA parity passed; debug snapshots saved", flush=True)
    del cpu_arrays, cuda_arrays
    gc.collect()
    torch.cuda.empty_cache()
    report["benchmark_method"] = dict(
        forward="inference_mode, separate act_inference/evaluate and their sequential sum; no Gaussian sampling",
        timing="One API call per sample; perf_counter wall time with CUDA synchronize. Event span includes host dispatch gaps. Setup/assertions excluded. Warmup and every raw sample recorded.",
        update="Native PPO.update: saved real maps, 4 rollout steps, 2 epochs x 4 minibatches, minibatch size equals listed batch. Includes Gaussian act/evaluate, loss/backward, gradient clip and Adam; rollout/GAE excluded from timing.",
        baseline="disabled and enabled both construct the same full perception policy and store the same raw maps; disabled skips glimpse/terrain K/V. This isolates compute, not a complete original Stage-1 memory baseline.",
        memory="CUDA PyTorch allocated/reserved only, excludes driver/context. Peak includes policy/inputs; PPO also includes rollout, gradients and warmed-up Adam states. Extra peak subtracts pre-call allocated. No Isaac/scanner/rendering cost.",
        scope="Untrained fixture, fixed source observations repeated to batch size. CPU one thread, float32 without TF32/autocast/compile. Not a training throughput or learned terrain-adaptation claim.")
    report["benchmarks"] = []
    for target in ("cpu", device):
        report["benchmarks"].extend(benchmark(state, source_obs, args, target))
    report["benchmark_optimizer_steps"] = sum(row.get("optimizer_steps_including_warmup", 0) for row in report["benchmarks"])
    report["environment"]["nvidia_smi_after"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True)
    report["artifacts"] = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "logs/pgmt_p21_acceptance/20261012_pmt_alignment_v3_r1")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/pgmt_p23_acceptance" / datetime.now().strftime("fusion_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=23505)
    parser.add_argument("--batches", type=int, nargs="+", default=[160, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ppo-iterations", type=int, default=5)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error(f"Run from {ROOT}")
    if args.warmup < 1 or args.iterations < 2 or args.ppo_iterations < 2 or any(n < 2 for n in args.batches):
        parser.error("Require warmup>=1, iterations/ppo-iterations>=2 and batches>=2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "scripts/validate_pgmt_terrain_glimpse.py", ROOT / "scripts/validate_pgmt_perception_ppo.py",
             ROOT / "motion_tracking_rl/networks/pgmt_actor_critic.py", ROOT / "motion_tracking_rl/networks/pgmt_terrain_glimpse.py",
             ROOT / "motion_tracking_rl/networks/pgmt_terrain_input.py", ROOT / "motion_tracking_rl/networks/layers/normalization.py",
             ROOT / "motion_tracking_rl/algorithms/ppo.py", ROOT / "motion_tracking_rl/storage/rollout_storage.py",
             ROOT / "motion_tracking_rl/pgmt_perception_contract.py", ROOT / "pmt_tasks/pgmt_perception.py",
             *[ROOT / name for name in REGRESSIONS]]
    report = dict(status="running", phase="P2.3 part 5 full-policy fusion acceptance", started_at=datetime.now().astimezone().isoformat(),
                  command=[sys.executable, *sys.argv], source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in paths},
                  limitations=["Synthetic proprio/history/commands/privileged observations; real saved terrain scans",
                               "Explicit checked old-weight fixture, not model_15000.pt migration or training",
                               "No Isaac, formal task entry or strict checkpoint resume; these remain P2.4/P2.5"])
    try:
        run(args, report, output)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Acceptance {report['status']}: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
