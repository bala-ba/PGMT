"""Reproducible P2.2 standalone acceptance; requires visible CUDA, not Isaac.

Run from PMT with the pmt Python environment. Outputs include the related pytest
suite, independent NumPy crop checks, CPU/CUDA comparisons, gradient evidence,
untrained-center diagnostics, synchronized benchmarks and an offline plot.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_tracking_rl.networks.pgmt_terrain_glimpse import (  # noqa: E402
    PGMTMaskedTerrainCrop, PGMTTerrainGlimpse, PGMTTerrainGlimpseInput,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_maps(source):
    contract_path = source / "terrain_observation_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["version"] == 3 and contract["units"] == "m"
    assert contract["height"] == "sensor_world_z - ground_world_z - offset_m"
    assert contract["normalization_location"] == "network"
    heights, masks, labels, sources = [], [], [], []
    names = [f"terrain_map_L{level}.npz" for level in (0, 5, 9)] + ["pmt_augmented_maps_L9.npz"]
    for name in names:
        path = source / name
        with np.load(path) as saved:
            height = saved["terrain_map"].astype(np.float32)
            mask = saved["terrain_valid" if "terrain_valid" in saved else "valid"].astype(bool)
            assert height.shape == mask.shape == (5, 21, 21)
            assert np.isfinite(height).all() and np.abs(height).max() <= 3
            if "physx_height" in saved:
                np.testing.assert_allclose(height, saved["physx_height"], atol=1e-4, rtol=0)
            families = saved["families"].astype(str).tolist()
            assert families == ["flat", "stairs", "slope", "boxes", "rough"]
        heights.append(height)
        masks.append(mask)
        labels.extend(f"{name}:{family}" for family in families)
        sources.append(dict(path=str(path), sha256=sha256(path)))
    return (torch.from_numpy(np.concatenate(heights)), torch.from_numpy(np.concatenate(masks)),
            labels, dict(contract=contract, contract_sha256=sha256(contract_path), files=sources))


def numpy_crop(height_m, valid, centers):
    """Independent scalar bilinear oracle in float64, without grid_sample."""
    height = np.clip(height_m.astype(np.float64), -3, 3) / 3
    n = len(height)
    output = np.full((n, 4, 5, 5), -1., dtype=np.float64)
    weights = np.zeros_like(output)
    point_valid = np.zeros_like(output, dtype=bool)
    for env in range(n):
        for patch in range(4):
            for row in range(5):
                for col in range(5):
                    x, y = centers[env, patch].astype(np.float64) + [(col - 2) / 10, (row - 2) / 10]
                    if not np.isfinite([x, y]).all() or max(abs(x), abs(y)) > 1:
                        continue
                    gx, gy = (x + 1) * 10, (y + 1) * 10
                    ix, iy = int(np.floor(gx)), int(np.floor(gy))
                    numerator = weight = 0.
                    for yy, wy in ((iy, 1 - (gy - iy)), (iy + 1, gy - iy)):
                        for xx, wx in ((ix, 1 - (gx - ix)), (ix + 1, gx - ix)):
                            if 0 <= xx < 21 and 0 <= yy < 21 and valid[env, yy, xx]:
                                weight += wx * wy
                                numerator += wx * wy * height[env, yy, xx]
                    weights[env, patch, row, col] = weight
                    if weight >= .5:
                        point_valid[env, patch, row, col] = True
                        output[env, patch, row, col] = numerator / max(weight, 1e-6)
    return output, weights, point_valid, point_valid.sum((-2, -1)) >= 13


def center_statistics(centers):
    centers = np.asarray(centers)
    i, j = np.triu_indices(4, 1)
    delta = np.abs(centers[:, i] - centers[:, j])
    distances = np.sqrt((delta ** 2).sum(-1))
    intersection = np.clip(.4 - delta, 0, None).prod(-1)
    iou = intersection / (.32 - intersection)
    return dict(environments=len(centers), centers=int(len(centers) * 4),
                coordinate_range_m=[float(centers.min()), float(centers.max())],
                per_patch_xy_mean_m=centers.mean(0).tolist(), per_patch_xy_std_m=centers.std(0).tolist(),
                boundary_coordinate_fraction=float((np.abs(centers) >= .79).mean()),
                pair_distance_quantiles_m=np.quantile(distances, [0, .05, .5, .95, 1]).tolist(),
                pair_distance_below_0_1m_fraction=float((distances < .1).mean()),
                patch_area_overlap_fraction=float((intersection > 0).mean()), mean_pair_iou=float(iou.mean()))


def map_cases(height, valid, seed):
    rng = torch.Generator().manual_seed(seed)
    cases = {"saved": valid.clone(), "extra_dropout_10pct": valid & (torch.rand(valid.shape, generator=rng) >= .1)}
    local = valid.clone()
    local[:, 7:14, 7:14] = False
    cases["central_hole_0_6m_span"] = local
    half = valid.clone()
    half[:, :, 11:] = False
    cases["map_boundary_x_positive"] = half
    cases["all_invalid"] = torch.zeros_like(valid)
    return {name: (torch.where(mask, height, -3.), mask) for name, mask in cases.items()}


def functional_audit(model, height, mask, labels, device):
    model = model.to(device)
    n = len(height)
    data = dict(terrain_map=height.to(device), terrain_valid=mask.to(device),
                history_token=torch.zeros(n, 1, 128, device=device), commands=torch.zeros(n, 8, 61, device=device))
    before = {key: value.clone() for key, value in data.items()}
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        result = model(**data, return_debug=True)
        repeated = model(**data)
    assert torch.equal(result.terrain_tokens, repeated.terrain_tokens)
    assert torch.equal(result.token_valid, repeated.token_valid)
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for key in data:
        assert torch.equal(data[key], before[key]), key
    tensors = dict(terrain_tokens=result.terrain_tokens, **result.debug._asdict())
    arrays = {key: value.cpu().numpy() for key, value in tensors.items()}
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert max(abs(arrays["center_xy_m"].ravel())) <= .8
    probes = {
        "selector": arrays["center_xy_m"],
        # Slightly inside the exact limit avoids float32 rounding at +/-1.
        "corners": np.tile([[-.799, -.799], [-.799, .799], [.799, -.799], [.799, .799]], (n, 1, 1)),
        "outside_scan": np.tile([[1.025, 0.], [-1.025, 0.], [0., 1.025], [0., -1.025]], (n, 1, 1)),
    }
    oracle_results = {}
    prepared = model.input_adapter(**data)
    with torch.no_grad():
        for name, xy in probes.items():
            centers = torch.tensor(xy, dtype=height.dtype, device=device)
            actual = model.crop(prepared, centers)
            expected = numpy_crop(height.numpy(), mask.numpy(), centers.cpu().numpy())
            observed = [actual.height.cpu().numpy(), actual.valid_weight.cpu().numpy(),
                        actual.point_valid.cpu().numpy(), actual.patch_valid.cpu().numpy()]
            errors = [float(np.max(np.abs(a - b))) for a, b in zip(observed[:2], expected[:2])]
            assert max(errors) < 1e-5, (device, name, errors)
            assert all(np.array_equal(a, b) for a, b in zip(observed[2:], expected[2:])), (device, name)
            oracle_results[name] = dict(points=int(observed[0].size), max_height_error=errors[0],
                                       max_weight_error=errors[1], masks_exact=True)
    report = dict(valid_patch_fraction=float(arrays["token_valid"].mean()),
                  valid_point_fraction=float(arrays["point_valid"].mean()),
                  per_map=[dict(label=label, coverage=arrays["coverage"][i].tolist(),
                                token_valid=arrays["token_valid"][i].tolist()) for i, label in enumerate(labels)],
                  centers=center_statistics(arrays["center_xy_m"]), numpy_oracle=oracle_results,
                  forward_repeat_exact=True, inputs_unchanged=True, rng_unchanged=True)
    return report, arrays


def gradient_audit(model, device):
    """Token loss through height-only path, plus masked finite differences."""
    model = model.to(device)
    with torch.no_grad():
        model.patch_encoder.mlp[0].weight[:, 50:].zero_()
    y, x = torch.meshgrid(torch.linspace(-1, 1, 21, device=device),
                          torch.linspace(-1, 1, 21, device=device), indexing="ij")
    height = (.7 + .6 * x - .4 * y)[None].requires_grad_()
    history = torch.zeros(1, 1, 128, device=device, requires_grad=True)
    commands = torch.zeros(1, 8, 61, device=device, requires_grad=True)
    result = model(height, torch.ones_like(height, dtype=torch.bool), history, commands)
    result.terrain_tokens.square().mean().backward()
    gradients = {name: p.grad.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
    gradients.update({name: value.grad.detach().cpu().numpy().copy()
                      for name, value in [("input_map", height), ("input_history", history), ("input_commands", commands)]})
    assert all(np.isfinite(value).all() and np.abs(value).sum() > 0 for value in gradients.values())
    # Double precision finite difference includes mask-weight denominator derivatives.
    axis = torch.linspace(-1, 1, 21, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    height = (.7 + .6 * x - .4 * y)[None]
    mask = torch.ones_like(height, dtype=torch.bool)
    mask[0, 11, 11] = False
    prepared = PGMTTerrainGlimpseInput()(height, mask, height.new_zeros(1, 1, 128), height.new_zeros(1, 8, 61))
    centers = torch.tensor([[[.023, .037], [-.153, .213], [.333, -.127], [-.457, -.223]]],
                           dtype=torch.float64, device=device, requires_grad=True)
    crop = PGMTMaskedTerrainCrop().to(device)
    crop(prepared, centers).height[:, :, 2, 2].sum().backward()
    numerical = torch.zeros_like(centers)
    with torch.no_grad():
        for patch in range(4):
            for dim in range(2):
                plus, minus = centers.detach().clone(), centers.detach().clone()
                plus[0, patch, dim] += 1e-6
                minus[0, patch, dim] -= 1e-6
                numerical[0, patch, dim] = (crop(prepared, plus).height[:, :, 2, 2].sum()
                                          - crop(prepared, minus).height[:, :, 2, 2].sum()) / 2e-6
    torch.testing.assert_close(centers.grad, numerical, atol=1e-9, rtol=1e-6)
    return dict(gradient_l1={name: float(np.abs(value).sum()) for name, value in gradients.items()},
                crop_finite_difference_max_abs_error=float((centers.grad - numerical).abs().max())), gradients


def compare_arrays(first, second, atol=1e-5, rtol=1e-4):
    errors = {}
    for name, a in first.items():
        b = second[name]
        if a.dtype == bool:
            assert np.array_equal(a, b), name
            errors[name] = 0.
        else:
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
            errors[name] = float(np.max(np.abs(a - b)))
    return errors


def distribution_audit(height, mask, device, seed):
    """Diagnostics, not learned foothold quality or a diversity constraint."""
    records = []
    for offset in range(5):
        torch.manual_seed(seed + offset)
        model = PGMTTerrainGlimpse().to(device)
        rng = torch.Generator().manual_seed(seed + 100 + offset)
        indices = torch.arange(1024) % len(height)
        for conditioning in ("zero", "synthetic_unit_normal"):
            history = torch.randn(1024, 1, 128, generator=rng) if conditioning != "zero" else torch.zeros(1024, 1, 128)
            commands = torch.randn(1024, 8, 61, generator=rng) if conditioning != "zero" else torch.zeros(1024, 8, 61)
            with torch.no_grad():
                result = model(height[indices].to(device), mask[indices].to(device),
                               history.to(device), commands.to(device), return_debug=True)
            records.append(dict(seed=seed + offset, conditioning=conditioning,
                                valid_patch_fraction=float(result.token_valid.float().mean()),
                                **center_statistics(result.debug.center_xy_m.cpu().numpy())))
    return records


def benchmark(state, height, mask, device, batches, warmup, iterations, cuda_inner_steps):
    records = []
    for n in batches:
        model = PGMTTerrainGlimpse().to(device)
        model.load_state_dict(state, strict=True)
        index = torch.arange(n) % len(height)
        data = dict(terrain_map=height[index].to(device), terrain_valid=mask[index].to(device),
                    history_token=torch.zeros(n, 1, 128, device=device, requires_grad=True),
                    commands=torch.zeros(n, 8, 61, device=device))
        for mode in ("inference", "inference_debug", "forward_backward"):
            def step():
                if mode == "forward_backward":
                    model.zero_grad(set_to_none=True)
                    data["history_token"].grad = None
                    result = model(**data)
                    (result.terrain_tokens.square() * result.token_valid[..., None]).mean().backward()
                else:
                    with torch.inference_mode():
                        model(**data, return_debug=mode == "inference_debug")

            for _ in range(warmup):
                step()
            model.zero_grad(set_to_none=True)
            data["history_token"].grad = None
            gc.collect()
            is_cuda = torch.device(device).type == "cuda"
            if is_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            durations, cuda_durations = [], []
            inner_steps = cuda_inner_steps if is_cuda else 1
            for _ in range(iterations):
                if is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                if is_cuda:
                    start_event.record()
                for _ in range(inner_steps):
                    step()
                if is_cuda:
                    end_event.record()
                    torch.cuda.synchronize(device)
                    cuda_durations.append(start_event.elapsed_time(end_event) / inner_steps)
                durations.append(1000 * (time.perf_counter() - start) / inner_steps)
            record = dict(device=str(device), batch=n, mode=mode, groups=iterations, calls_per_group=inner_steps,
                          median_ms=float(np.median(durations)), p95_ms=float(np.quantile(durations, .95)))
            if is_cuda:
                peak = torch.cuda.max_memory_allocated(device)
                record.update(baseline_allocated_MiB=baseline / 2**20, peak_allocated_MiB=peak / 2**20,
                              extra_peak_MiB=(peak - baseline) / 2**20,
                              peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 2**20,
                              cuda_event_median_ms=float(np.median(cuda_durations)),
                              cuda_event_p95_ms=float(np.quantile(cuda_durations, .95)))
            records.append(record)
        del model, data
        print(f"Benchmark complete: {device}, batch={n}", flush=True)
    return records


def save_plot(height, mask, centers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#a0a0a0")
    families = ["flat", "stairs", "slope", "boxes", "rough"]
    rows = ["L0", "L5", "L9", "L9 augmented"]
    for i, ax in enumerate(axes.flat):
        shown = np.ma.masked_where(~mask[i], height[i])
        ax.imshow(shown, extent=(-1.05, 1.05, -1.05, 1.05), origin="lower", cmap=cmap, vmin=-3, vmax=3)
        for j, (x, y) in enumerate(centers[i]):
            color = ["#ff595e", "#ffca3a", "#8ac926", "#6ac8ff"][j]
            ax.add_patch(Rectangle((x - .2, y - .2), .4, .4, fill=False, edgecolor=color, linewidth=1.2))
            ax.plot(x, y, "+", color=color)
        ax.set_title(f"{rows[i // 5]} / {families[i % 5]}")
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel="torso-yaw x (m)", ylabel="y (m)")
    fig.suptitle("Untrained Terrain-Glimpse | sensor-ground height, -3..3 m | gray = invalid\n"
                 "Colored squares: 0.4 m sampling spans; centers are not learned footholds")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args, report, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run outside a sandbox that hides /dev/nvidia*; no silent GPU skips.")
    device = torch.device(args.device)
    assert device.type == "cuda"
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["environment"] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        cuda_build=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device),
        cpu_threads=torch.get_num_threads(), float32_matmul_precision=torch.get_float32_matmul_precision(),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True))
    suites = ["tests/test_pgmt_terrain_glimpse.py", "tests/test_pgmt_perception.py",
              "tests/test_pgmt_rl.py", "tests/test_normalization_resume.py"]
    command = [sys.executable, "-m", "pytest", "-q", *suites, f"--junitxml={output / 'pytest.xml'}"]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "pytest.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    xml = ET.parse(output / "pytest.xml").getroot()
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in xml.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    assert counts["skipped"] == 0, counts
    report["pytest"] = dict(command=command, **counts)
    print(f"Related regression passed: {counts}", flush=True)
    height, mask, labels, report["inputs"] = load_maps(args.source)
    torch.manual_seed(args.seed)
    reference = PGMTTerrainGlimpse()
    state = {key: value.clone() for key, value in reference.state_dict().items()}
    report["config"] = dict(selector=asdict(reference.selector.config), crop=asdict(reference.crop.config),
                           encoder=asdict(reference.patch_encoder.config), dtype="float32",
                           num_parameters=sum(p.numel() for p in reference.parameters()))

    def fresh():
        model = PGMTTerrainGlimpse()
        model.load_state_dict(state, strict=True)
        return model

    report["functional"] = {}
    for name, (case_height, case_mask) in map_cases(height, mask, args.seed).items():
        cpu, ca = functional_audit(fresh(), case_height, case_mask, labels, "cpu")
        gpu, ga = functional_audit(fresh(), case_height, case_mask, labels, device)
        parity = compare_arrays(ca, ga)
        if name == "all_invalid":
            assert not ga["token_valid"].any() and np.all(ga["patch_height"] == -1)
        report["functional"][name] = dict(cpu=cpu, cuda=gpu, cpu_cuda_max_abs_errors=parity)
        if name == "saved":
            np.savez_compressed(output / "real_maps_debug.npz", terrain_map_m=height.numpy(),
                                terrain_valid=mask.numpy(), labels=np.array(labels), **ga)
            save_plot(height.numpy(), mask.numpy(), ga["center_xy_m"], output / "real_maps_glimpses.png")
        print(f"CPU/CUDA and NumPy checks passed: {name}", flush=True)
    cpu_grad, ca = gradient_audit(fresh(), "cpu")
    gpu_grad, ga = gradient_audit(fresh(), device)
    report["gradients"] = dict(cpu=cpu_grad, cuda=gpu_grad, cpu_cuda_max_abs_errors=compare_arrays(ca, ga, atol=1e-6))
    report["center_distribution"] = distribution_audit(height, mask, device, args.seed)
    print("Gradient and center-distribution checks passed", flush=True)
    report["benchmark_method"] = dict(warmup=args.warmup, groups=args.iterations, cuda_calls_per_group=args.cuda_inner_steps,
        timing="perf_counter wall time per call, synchronize after each group; CUDA Event group span also reported; CPU groups contain one call. Event span includes host dispatch gaps, not pure kernel time. First allocator-cold group retained.",
        scope="Standalone float32 glimpse; no Isaac/scanner/IFM/PPO/optimizer. Forward_backward includes history input gradients.",
        memory="PyTorch live allocated peak includes parameters/inputs and library workspace; extra peak subtracts baseline; excludes driver/context and optimizer.",
        input="Repeated saved version-3 maps, zero history and normalized commands; no augmentation resampling.")
    report["benchmarks"] = benchmark(state, height, mask, "cpu", args.batches, args.warmup, args.iterations, args.cuda_inner_steps)
    report["benchmarks"] += benchmark(state, height, mask, device, args.batches, args.warmup, args.iterations, args.cuda_inner_steps)
    report["artifacts"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "logs/pgmt_p21_acceptance/20261012_pmt_alignment_v3_r1")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/pgmt_p22_acceptance" / datetime.now().strftime("full_%Y%m%d_%H%M%S"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2245)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 160, 1024, 8192])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cuda-inner-steps", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 2 or args.cuda_inner_steps < 1 or any(n < 1 for n in args.batches):
        parser.error("Require warmup>=1, iterations>=2, cuda-inner-steps>=1 and positive batch sizes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "motion_tracking_rl/networks/pgmt_terrain_glimpse.py",
               ROOT / "motion_tracking_rl/networks/pgmt_terrain_input.py", ROOT / "tests/test_pgmt_terrain_glimpse.py"]
    report = dict(status="running", phase="P2.2 standalone acceptance", started_at=datetime.now().isoformat(),
                  seed=args.seed, command=[sys.executable, *sys.argv], source_sha256={str(p): sha256(p) for p in sources})
    try:
        run(args, report, output)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now().isoformat()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Acceptance {report['status']}: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
