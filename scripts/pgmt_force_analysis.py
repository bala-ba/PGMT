"""Offline comparison of paired PGMT physics-step force recordings (no Isaac)."""
import csv
import json
from pathlib import Path

import numpy as np


VECTOR_FIELDS = ("net_forces_w", "terrain_forces_w", "foot_pos_w", "foot_vel_w", "reference_pos_w")


def load_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "training_status.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"Run did not complete: {directory}")
    contract = metadata.get("force_recording", {})
    if contract.get("schema_version") != 1:
        raise ValueError(f"Missing physics-step force recording contract: {directory}")
    with np.load(directory / "foot_forces.npz", allow_pickle=False) as archive:
        data = dict(archive)
    n = len(data["time_s"])
    if n == 0:
        raise ValueError(f"Empty force recording: {directory}")
    for key in VECTOR_FIELDS:
        if data[key].shape != (n, 2, 3):
            raise ValueError(f"Invalid {key} shape in {directory}")
    for key in ("reference_contact", "reference_valid"):
        if data[key].shape != (n, 2) or not np.isin(data[key], [0, 1]).all():
            raise ValueError(f"Invalid {key} labels in {directory}")
    keys = list(zip(data["episode"].tolist(), data["frame"].tolist(), data["substep"].tolist()))
    if len(set(keys)) != n:
        raise ValueError(f"Repeated episode/frame/substep in {directory}")
    dt = metadata["physics_dt"]
    for episode in np.unique(data["episode"]):
        times = data["time_s"][data["episode"] == episode]
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), dt, atol=1e-7, rtol=0):
            raise ValueError(f"Nonconsecutive physics samples in {directory}")
    if len(np.unique(data["terrain"])) != 1:
        raise ValueError("One terrain per comparison run is required; disable alternating / GUI switching")
    return dict(directory=str(directory), metadata=metadata, data=data,
                key_index={key: i for i, key in enumerate(keys)})


def validate_pairing(runs):
    if "flat" not in runs or len(runs) < 2:
        raise ValueError("Comparison requires flat and at least one other terrain")
    base = runs["flat"]["metadata"]
    for name, run in runs.items():
        meta = run["metadata"]
        if str(run["data"]["terrain"][0]) != name:
            raise ValueError(f"Terrain label does not match recording: {name}")
        for key in ("source_sha256", "physics_dt", "control_dt"):
            if meta.get(key) is None or meta[key] != base[key]:
                raise ValueError(f"Unpaired {key}: {name}")
        for key in ("motion_sha256", "terrain_manifest_sha256", "seed", "spawn_offset_x",
                    "force_semantics", "reference_sampling", "sampling", "foot_names",
                    "decimation", "contact_config", "physics_buffer_capacities"):
            if key not in meta["force_recording"] or meta["force_recording"][key] != base["force_recording"][key]:
                raise ValueError(f"Unpaired {key}: {name}")
        for key in ("patch_id", "direction"):
            if meta["arguments"][key] != base["arguments"][key]:
                raise ValueError(f"Unpaired playback {key}: {name}")
        if meta["force_recording"]["spawn_offset_x"] is None:
            raise ValueError("Use an explicit common spawn_offset_x for paired trials")
        for key in ("observation_noise", "random_pushes", "random_dynamics", "curriculum_updates", "action_mode"):
            if key not in meta or meta[key] != base[key]:
                raise ValueError(f"Unpaired inference setting {key}: {name}")


def match_events(actual_times, reference_times, tolerance_s=.3):
    """Order-preserving one-to-one matching; missing events never shift step IDs."""
    n, m = len(actual_times), len(reference_times)
    cost = np.zeros((n + 1, m + 1))
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[:, 0], cost[0, :] = np.arange(n + 1), np.arange(m + 1)
    choice[1:, 0], choice[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            error = abs(actual_times[i - 1] - reference_times[j - 1])
            options = [cost[i - 1, j - 1] + error / tolerance_s if error <= tolerance_s else np.inf,
                       cost[i - 1, j] + 1, cost[i, j - 1] + 1]
            choice[i, j] = np.argmin(options)
            cost[i, j] = min(options)
    pairs, i, j = [], n, m
    while i or j:
        action = choice[i, j]
        if action == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def support_onsets(force_z, valid, dt, threshold=10., min_off_s=.04, min_on_s=.01):
    """Diagnostic onset: confirmed after min_on_s, timestamped at first crossing.

    Initial loaded feet and missing measurements cannot create a touchdown.
    This is independent of the unchanged production 50 Hz contact state machine.
    """
    min_off = max(1, int(np.ceil(min_off_s / dt - 1e-9)))
    min_on = max(1, int(np.ceil(min_on_s / dt - 1e-9)))
    off, on, eligible, start = 0, 0, False, 0
    result = []
    for i, (z, ok) in enumerate(zip(force_z, valid)):
        if not ok or not np.isfinite(z):
            off, on, eligible = 0, 0, False
        elif z <= threshold:
            off, on, eligible = off + 1, 0, False
        else:
            if on == 0:
                eligible, start = off >= min_off, i
            off, on = 0, on + 1
            if eligible and on == min_on:
                result.append(start)
    return result


def analyze_events(run, tolerance_s=.3, pre_s=.1, post_s=.15):
    data, meta = run["data"], run["metadata"]
    dt = meta["physics_dt"]
    config = meta["force_recording"]["contact_config"]
    output, references = [], []
    for episode in np.unique(data["episode"]):
        ids = np.flatnonzero(data["episode"] == episode)
        times = data["time_s"][ids]
        for foot, side in enumerate(("L", "R")):
            force = data["terrain_forces_w"][ids, foot]
            finite = np.isfinite(force).all(-1)
            actual = support_onsets(force[:, 2], finite, dt,
                threshold=config["support_threshold_n"],
                min_off_s=config["touchdown_min_non_support_steps"] * meta["control_dt"])
            contact = data["reference_contact"][ids, foot].astype(bool)
            valid = data["reference_valid"][ids, foot].astype(bool)
            ref = np.flatnonzero(contact[1:] & ~contact[:-1] & valid[1:] & valid[:-1]) + 1
            # A held reference changes at the BEGINNING of its control interval.
            ref_times = times[ref] - dt
            pairs = match_events(times[actual], ref_times, tolerance_s)
            matching = {a: r for a, r in pairs}
            candidates = np.abs(times[actual, None] - ref_times[None, :]) <= tolerance_s
            for r, index in enumerate(ref):
                references.append(dict(episode=int(episode), foot=side,
                    reference_frame=int(data["frame"][ids[index]]), time_s=float(ref_times[r]),
                    candidate_actual_count=int(candidates[:, r].sum()),
                    matched=r in matching.values()))
            for a, index in enumerate(actual):
                r = matching.get(a)
                lo, hi = times[index] - pre_s, times[index] + post_s
                window = (times >= lo - 1e-8) & (times <= hi + 1e-8)
                complete = bool(times[0] <= lo + 1e-8 and times[-1] >= hi - 1e-8 and finite[window].all())
                target_index = ids[ref[r]] if r is not None else None
                error = (data["foot_pos_w"][ids[index], foot] - data["reference_pos_w"][target_index, foot]
                         if target_index is not None else None)
                if error is not None and not np.isfinite(error).all():
                    error = None
                output.append(dict(episode=int(episode), foot=side, sample_index=int(ids[index]),
                    time_s=float(times[index]), frame=int(data["frame"][ids[index]]),
                    reference_frame=int(data["frame"][target_index]) if target_index is not None else None,
                    ambiguous_match=bool(candidates[:, r].sum() > 1 or candidates[a].sum() > 1) if r is not None else None,
                    delay_s=float(times[index] - ref_times[r]) if r is not None else None,
                    window_complete=complete,
                    peak_terrain_fz_n=float(force[window, 2].max()) if complete else None,
                    upward_normal_impulse_ns=float(np.maximum(force[window, 2], 0).sum() * dt) if complete else None,
                    ankle_landing_error_w_m=error.tolist() if error is not None else None))
    return dict(actual=output, reference=references)


def compare(runs, *, skip_initial_s=.1, tolerance_s=.3):
    validate_pairing(runs)
    common = set.intersection(*(set(run["key_index"]) for run in runs.values()))
    keys = sorted(k for k in common if runs["flat"]["data"]["time_s"][runs["flat"]["key_index"][k]] > skip_initial_s)
    if not keys:
        raise ValueError("No common physics samples after the initial exclusion")
    indices = {name: np.array([run["key_index"][k] for k in keys]) for name, run in runs.items()}
    base = runs["flat"]["data"]
    dt = runs["flat"]["metadata"]["physics_dt"]
    metrics = []
    for name, run in runs.items():
        data, ids = run["data"], indices[name]
        flat_ids = indices["flat"]
        for field in ("reference_contact", "reference_valid"):
            if not np.array_equal(data[field][ids], base[field][flat_ids]):
                raise ValueError(f"Mismatched held reference labels: {name}")
        for foot, side in enumerate(("L", "R")):
            force, flat = data["terrain_forces_w"][ids, foot], base["terrain_forces_w"][flat_ids, foot]
            valid = np.isfinite(force).all(-1) & np.isfinite(flat).all(-1)
            values, baseline = force[valid], flat[valid]
            delta = data["foot_pos_w"][ids, foot] - data["reference_pos_w"][ids, foot]
            pose_ok = np.isfinite(delta).all(-1)
            contact_ok = np.isfinite(force).all(-1) & data["reference_valid"][ids, foot].astype(bool)
            actual_support = force[:, 2] > run["metadata"]["force_recording"]["contact_config"]["support_threshold_n"]
            mismatch = actual_support != data["reference_contact"][ids, foot].astype(bool)
            metrics.append(dict(terrain=name, foot=side, common_samples=len(ids),
                valid_force_samples=int(valid.sum()), common_duration_s=len(ids) * dt,
                mean_terrain_fz_n=float(values[:, 2].mean()) if len(values) else None,
                peak_terrain_fz_n=float(values[:, 2].max()) if len(values) else None,
                peak_terrain_xy_n=float(np.linalg.norm(values[:, :2], axis=-1).max()) if len(values) else None,
                support_mismatch_fraction=float(mismatch[contact_ok].mean()) if contact_ok.any() else None,
                upward_normal_impulse_ns=float(np.maximum(values[:, 2], 0).sum() * dt) if valid.all() else None,
                rms_force_difference_from_flat_n=float(np.sqrt(np.mean(np.sum((values-baseline)**2, axis=-1)))) if len(values) else None,
                ankle_reference_rmse_m=float(np.sqrt(np.mean(np.sum(delta[pose_ok]**2, axis=-1)))) if pose_ok.any() else None))
    events = {name: analyze_events(run, tolerance_s=tolerance_s) for name, run in runs.items()}
    return dict(schema_version=1, skip_initial_s=skip_initial_s, match_tolerance_s=tolerance_s,
        force_semantics="normal_only; horizontal force is not friction; world Z is not surface normal on slopes",
        reference_semantics="same motion command and checkpoint; physical motion can diverge on terrain",
        time_semantics="force is a physics-interval sample at its end; reference held from interval start",
        position_semantics="ankle link origin, not sole/contact point; landing error uses matched reference touchdown pose",
        event_window_s=[-.1, .15], common_frame_range=[keys[0][1], keys[-1][1]],
        metrics=metrics, events=events,
        event_summary={name: dict(actual=len(e["actual"]), reference=len(e["reference"]),
            unmatched_actual=sum(t["reference_frame"] is None for t in e["actual"]),
            unmatched_reference=sum(not t["matched"] for t in e["reference"]),
            ambiguous_matches=sum(t["ambiguous_match"] is True for t in e["actual"])) for name, e in events.items()},
        coverage={name: dict(samples=len(run["data"]["time_s"]),
            completed_episodes=run["metadata"].get("completed_episodes"),
            max_frame=int(run["data"]["frame"].max()), directory=run["directory"]) for name, run in runs.items()})


def write_report(runs, output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["metrics"][0]))
        writer.writeheader()
        writer.writerows(report["metrics"])
    with (output / "touchdowns.csv").open("w", newline="") as stream:
        rows = [dict(terrain=name, **event) for name, events in report["events"].items() for event in events["actual"]]
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Plot first episode only; statistics/events retain all episodes separately.
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, constrained_layout=True)
    for name, run in runs.items():
        d = run["data"]
        ids = np.flatnonzero(d["episode"] == np.min(d["episode"]))
        t = d["frame"][ids] * run["metadata"]["control_dt"] + (d["substep"][ids] + 1) * run["metadata"]["physics_dt"]
        for foot, side in enumerate(("L", "R")):
            f = d["terrain_forces_w"][ids, foot]
            axes[0, foot].plot(t, f[:, 2], label=name, lw=.8)
            axes[1, foot].plot(t, np.linalg.norm(f[:, :2], axis=-1), lw=.8)
            axes[2, foot].plot(t, d["foot_pos_w"][ids, foot, 2] - d["reference_pos_w"][ids, foot, 2], lw=.8)
            axes[0, foot].set_title(f"{side} foot / first episode")
            axes[2, foot].set_xlabel("Reference time + physics substep (s)")
    for row, label in enumerate(("Terrain normal-force world Z (N)", "Terrain normal-force |XY| (N)", "Ankle Z - held reference Z (m)")):
        for ax in axes[row]:
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.savefig(output / "reference_aligned.png", dpi=160)
    plt.close(fig)
    # Align only events assigned to the SAME reference landing, never nth touchdown.
    maps = {name: {(e["episode"], e["foot"], e["reference_frame"]): e for e in events["actual"]
                   if e["reference_frame"] is not None and e["window_complete"]}
            for name, events in report["events"].items()}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    if common:
        selected = common[:8]
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.7 * len(selected)), squeeze=False, constrained_layout=True)
        for ax, key in zip(axes[:, 0], selected):
            for name, run in runs.items():
                event, d = maps[name][key], run["data"]
                t = d["time_s"] - event["time_s"]
                ids = (d["episode"] == key[0]) & (t >= -.1 - 1e-8) & (t <= .15 + 1e-8)
                ax.plot(t[ids], d["terrain_forces_w"][ids, ("L", "R").index(key[1]), 2], label=name)
            ambiguous = any(maps[name][key]["ambiguous_match"] for name in maps)
            ax.set(title=f"Episode {key[0]} / {key[1]} / reference landing frame {key[2]}"
                   + (" [AMBIGUOUS time match]" if ambiguous else ""),
                   xlabel="Time from measured touchdown (s)", ylabel="Terrain normal Fz (N)")
            ax.axvline(0, color="k", lw=.6)
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(output / "touchdown_aligned.png", dpi=160)
        plt.close(fig)
    else:
        (output / "touchdown_aligned.png").unlink(missing_ok=True)
    report["common_complete_touchdowns"] = len(common)
    report["common_unambiguous_complete_touchdowns"] = sum(
        not any(m[key]["ambiguous_match"] for m in maps.values()) for key in common)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="TERRAIN=DIRECTORY")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip_initial_s", type=float, default=.1)
    parser.add_argument("--match_tolerance_s", type=float, default=.3)
    args = parser.parse_args()
    if not np.isfinite(args.skip_initial_s) or args.skip_initial_s < 0 or not np.isfinite(args.match_tolerance_s) or args.match_tolerance_s <= 0:
        parser.error("Invalid exclusion or matching tolerance")
    pairs = [value.split("=", 1) for value in args.run]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        parser.error("Use distinct TERRAIN=DIRECTORY inputs")
    runs = {name: load_run(path) for name, path in pairs}
    report = compare(runs, skip_initial_s=args.skip_initial_s, tolerance_s=args.match_tolerance_s)
    write_report(runs, args.output, report)
    print(json.dumps({"output": str(Path(args.output).resolve()), "coverage": report["coverage"],
                      "common_complete_touchdowns": report["common_complete_touchdowns"]}, indent=2))


if __name__ == "__main__":
    main()
