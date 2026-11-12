"""Offline PGMT terrain curves; no simulator or training/checkpoint writes.

Episode lengths are exposure/outcome estimates, not recovered terminal lengths.
The source run must be a contiguous, single-process run starting at iteration 0.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pgmt-analysis-mpl")
import numpy as np
import torch

OUTCOMES = ("success", "failure", "map_exit", "timeout", "incomplete_motion", "other_truncation")
LABELS = {"flat": "平地 flat", "stairs": "楼梯 stairs", "slope": "斜坡 slope",
          "boxes": "箱体 boxes", "rough": "粗糙 rough"}
COLORS = ("#2563eb", "#db6b18", "#169b78", "#a64bb3", "#cf4660")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_progress(run, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved.get("distributed") is None, "This report expects a single-process run"
    course = saved["environment_state_dict"]["curriculum"]
    contract = saved["training_contract"]
    env_cfg = contract["runtime"]["environment"]
    families = course["identity"]["families"]
    assert len(families) == 5
    n = saved["next_iteration"]
    steps = int(contract["num_steps_per_env"])
    dt = env_cfg["sim"]["dt"] * env_cfg["decimation"]
    max_steps = math.ceil(env_cfg["episode_length_s"] / dt)
    levels = course["counts"].shape[1]
    counts = np.zeros((n, len(families), levels, len(OUTCOMES)), dtype=np.int64)
    population = np.zeros((n, len(families), levels), dtype=np.int64)
    means = np.zeros((n, len(families)))
    times = np.zeros((n, 2))
    cumulative = np.zeros(counts.shape[1:], dtype=np.int64)
    before = cumulative.copy()
    keys = [(f, l, f"Curriculum/{name}/L{l}", f"OutcomeWindow/{name}/L{l}")
            for f, name in enumerate(families) for l in range(levels)]
    done_counts = np.zeros(n, dtype=np.int64)
    tag_set = set()
    progress_hash = hashlib.sha256()
    total_lines = 0
    with (run / "progress.jsonl").open("rb") as file:
        for i, line in enumerate(file):
            progress_hash.update(line)
            record = json.loads(line)
            assert i < n and record["iteration"] == i and record["completed_iteration"] == i + 1
            assert record["tensorboard_iteration"] == i
            telemetry = record["telemetry"]
            tag_set.update(telemetry)
            assert telemetry["Window/transitions"] == steps * env_cfg["scene"]["num_envs"]
            times[i] = record["collection_s"], record["update_s"]
            for f, l, prefix, window in keys:
                population[i, f, l] = telemetry[f"{prefix}/env_count"]
                for o, name in enumerate(OUTCOMES):
                    counts[i, f, l, o] = telemetry[f"{window}/{name}"]
                    cumulative[f, l, o] = telemetry[f"{prefix}/{name}"]
                assert telemetry[f"{prefix}/outcomes"] == cumulative[f, l].sum()
                if cumulative[f, l].sum():
                    assert abs(telemetry[f"{prefix}/success_fraction_of_outcomes"] -
                               cumulative[f, l, 0] / cumulative[f, l].sum()) < 1e-7
            assert np.array_equal(cumulative - before, counts[i])
            assert (counts[i] >= 0).all()
            before[:] = cumulative
            for f, name in enumerate(families):
                means[i, f] = telemetry[f"Curriculum/{name}/mean_level"]
                expected = population[i, f] @ np.arange(levels) / population[i, f].sum()
                assert abs(means[i, f] - expected) < 2e-6
            done_counts[i] = record["reward_group_audit"]["done_events"]
            assert cumulative.sum() == done_counts[i]
            total_lines = i + 1
            if total_lines % 5000 == 0:
                print(f"JSON: {total_lines}/{n} iterations verified", flush=True)
    assert total_lines == n
    assert np.array_equal(cumulative, course["counts"].numpy())
    assert population.sum((1, 2)).min() == population.sum((1, 2)).max() == env_cfg["scene"]["num_envs"]
    assert (population.sum(2) == population[0].sum(1)).all(), "Family population changed"
    assert not [k for k in tag_set if "length" in k.lower() or "duration" in k.lower()]
    # A success moves one level up except at L9; a failure moves down except at L0.
    arrivals = np.zeros_like(population)
    departures = np.zeros_like(population)
    arrivals[:, :, 1:] += counts[:, :, :-1, 0]
    departures[:, :, :-1] += counts[:, :, :-1, 0]
    arrivals[:, :, :-1] += counts[:, :, 1:, 1]
    departures[:, :, 1:] += counts[:, :, 1:, 1]
    flow = arrivals - departures
    initial_population = population[0] - flow[0]
    assert (initial_population >= 0).all()
    previous_population = np.concatenate([initial_population[None], population[:-1]])
    assert np.array_equal(population - previous_population, flow), "Unexplained curriculum/reset changes"
    # Pre-step exposure is integrated from two post-step boundary snapshots.
    exposure = steps * (previous_population + population) / 2
    # Every transition can displace the trapezoid integral by at most steps/2.
    exposure_error_bound = steps / 2 * (arrivals + departures)
    assert np.array_equal(exposure.sum(2), np.broadcast_to(steps * population[0].sum(1), (n, 5)))
    metadata = dict(families=families, iterations=n, num_envs=env_cfg["scene"]["num_envs"],
                    steps_per_iteration=steps, dt=dt, max_episode_steps=max_steps,
                    episode_length_s=env_cfg["episode_length_s"], outcomes=list(OUTCOMES),
                    progress_sha256=progress_hash.hexdigest(), checkpoint_sha256=digest(checkpoint),
                    motion_lengths=sorted(set(course["identity"]["motion_lengths"])),
                    family_env_counts=population[0].sum(1).tolist(),
                    final_outcome_totals=cumulative.sum((0, 1)).tolist(),
                    validations=dict(contiguous_iterations=True, nonnegative_window_counts=True,
                        cumulative_differences_match=True, final_counts_match_checkpoint=True,
                        done_audit_matches_outcomes=True, population_flow_matches=True,
                        constant_family_population=True, mean_levels_match_counts=True))
    return dict(counts=counts, population=population, initial_population=initial_population,
                previous_population=previous_population, mean_level=means, exposure=exposure,
                exposure_error_bound=exposure_error_bound, compute_times=times), metadata


def read_event_clocks(run, n):
    """Read selected TFRecords without retaining millions of unrelated scalars.

    Validate framing throughout, and CRC32C for the selected records. Parse all
    records in a small first section to inventory tags, then parse matching tags.
    """
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
    from tensorboard.util import tensor_util
    wall = np.full(n, np.nan)
    global_length = np.full(n, np.nan)
    tags = set()
    inventory_tags = set()
    records = 0
    selected_records = 0
    paths = sorted(run.glob("events.out.tfevents.*"))
    assert paths
    for path in paths:
        with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            offset = 0
            while offset < len(data):
                assert offset + 12 <= len(data), "Truncated TFRecord header"
                size = struct.unpack_from("<Q", data, offset)[0]
                end = offset + 12 + size
                assert end + 4 <= len(data), "Truncated TFRecord payload"
                payload = data[offset + 12:end]
                wanted = (b"Perf/total_fps" in payload or b"mean_episode_length" in payload)
                if wanted or records < 6000:
                    assert struct.unpack_from("<I", data, offset + 8)[0] == masked_crc32c(data[offset:offset + 8])
                    assert struct.unpack_from("<I", data, end)[0] == masked_crc32c(payload)
                    event = Event.FromString(payload)
                    for value in event.summary.value:
                        if records < 6000:
                            inventory_tags.add(value.tag)
                        if "mean_episode_length" in value.tag:
                            tags.add(value.tag)
                        if value.tag == "Perf/total_fps":
                            assert 0 <= event.step < n and np.isnan(wall[event.step])
                            wall[event.step] = event.wall_time
                        elif value.tag == "Train/mean_episode_length":
                            assert 0 <= event.step < n and np.isnan(global_length[event.step])
                            global_length[event.step] = (value.simple_value if value.HasField("simple_value")
                                else float(tensor_util.make_ndarray(value.tensor)))
                    selected_records += 1
                offset = end + 4
                records += 1
                if records % 5000000 == 0:
                    print(f"TensorBoard: {records:,} records scanned", flush=True)
    assert np.isfinite(wall).all() and (np.diff(wall) > 0).all()
    assert np.isfinite(global_length).all()
    assert tags == {"Train/mean_episode_length", "Train/mean_episode_length/time"}, tags
    return wall, global_length, dict(event_files=[str(p) for p in paths], record_count=records,
        checked_selected_records=selected_records, length_tags=sorted(tags),
        inventory_tag_count=len(inventory_tags))


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b != 0)


def summarize_window(data, start, stop, max_steps, by_level=False):
    counts = data["counts"][start:stop].sum(0)
    exposure = data["exposure"][start:stop].sum(0)
    exposure_error = data["exposure_error_bound"][start:stop].sum(0)
    p0, p1 = data["previous_population"][start], data["population"][stop - 1]
    if not by_level:
        counts = counts.sum(1)
        exposure = exposure.sum(1)
        exposure_error = np.zeros_like(exposure)  # family population never changes
        p0, p1 = p0.sum(1), p1.sum(1)
    ended = counts.sum(-1)
    estimate = ratio(exposure, ended)
    # Exact completed-episode total = true exposure + age_at_start - age_at_end.
    # max_steps is a conservative bound on each still-active episode's age.
    lower = np.maximum(0, ratio(exposure - exposure_error - max_steps * p1, ended))
    upper = np.minimum(max_steps, ratio(exposure + exposure_error + max_steps * p0, ended))
    # Every episode at run start has age zero, which tightens that boundary.
    if start == 0:
        upper = np.minimum(max_steps, ratio(exposure + exposure_error, ended))
    return dict(counts=counts, ended=ended, success_rate=ratio(counts[..., 0], ended),
        course_success_rate=ratio(counts[..., 0], counts[..., :2].sum(-1)),
        episode_steps_estimate=estimate, episode_steps_bound_low=lower, episode_steps_bound_high=upper,
        exposure=exposure, exposure_error_bound=exposure_error,
        mean_level=data["mean_level"][start:stop].mean(0))


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_report(data, meta, output, window, last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 18, "axes.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#aeb9c7", "axes.labelcolor": "#26374c", "text.color": "#26374c",
        "xtick.color": "#526278", "ytick.color": "#526278", "svg.fonttype": "none"})
    n, families, dt = meta["iterations"], meta["families"], meta["dt"]
    assert 0 < last <= n and window > 0
    blocks = [(a, min(a + window, n)) for a in range(0, n, window)]
    stats = [summarize_window(data, a, b, meta["max_episode_steps"]) for a, b in blocks]
    final = summarize_window(data, n - last, n, meta["max_episode_steps"], by_level=True)
    final_family = summarize_window(data, n - last, n, meta["max_episode_steps"])
    x = np.array([(data["elapsed_hours"][a] + data["elapsed_hours"][b - 1]) / 2 for a, b in blocks])
    figure_names = []

    def axes(title, ylabel, time_axis=False, estimate=False):
        fig, ax = plt.subplots(figsize=(13.5, 7.2), layout=None)
        fig.subplots_adjust(left=.085, right=.91, bottom=.23, top=.76)
        fig.text(.085, .94, title, fontsize=19, weight="bold")
        subtitle = (f"每 {window} 轮独立汇总 · 8192 环境 · 五类地形" if time_axis
                    else f"最后 {last:,} 轮：iteration {n-last:,}–{n-1:,} · 按出生地形及等级统计")
        fig.text(.085, .895, subtitle, fontsize=11, color="#63748a")
        ax.set_ylabel(ylabel)
        ax.grid(axis="both", color="#e5eaf1", linewidth=.8)
        ax.set_axisbelow(True)
        if time_axis:
            h = np.r_[0., data["elapsed_hours"]]
            iterations = np.arange(n + 1)
            secondary = ax.secondary_xaxis("top", functions=(
                lambda value: np.interp(value, h, iterations),
                lambda value: np.interp(value, iterations, h)))
            secondary.set_xticks(np.arange(0, n + 1, 5000))
            secondary.set_xticklabels([f"{v:,}" for v in range(0, n + 1, 5000)])
            secondary.set_xlabel("已完成 PPO 轮数", labelpad=8, fontsize=10)
            secondary.tick_params(labelsize=10)
            ax.set_xlabel("实际训练时间（小时）", labelpad=10)
            ax.set_xlim(0, data["elapsed_hours"][-1])
        else:
            ax.set_xlabel("出生时课程难度等级", labelpad=10)
            ax.set_xticks(range(10), [f"L{i}" for i in range(10)])
            ax.set_xlim(-.15, 9.15)
        foot = ("估算：累计环境控制步数 ÷ 已结束回合数；分等级使用环境数量快照的梯形积分。\n"
                "未记录分组终止时长，无法精确恢复；窗口边界未结束回合会影响估算，详见计算说明。"
                if estimate else "成功率：成功回合数 ÷ 全部已结束回合数；包含失败及中性截断，按计数汇总后求比值。")
        return fig, ax, foot

    def save(fig, ax, index, stem, foot):
        ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(-.01, 1.235),
                  frameon=False, fontsize=10, handlelength=2.3, columnspacing=1.4)
        fig.text(.085, .065, foot, fontsize=10, color="#63748a", linespacing=1.55)
        name = f"{index:02d}_{stem}"
        fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
        fig.savefig(output / f"{name}.svg", facecolor="white")
        plt.close(fig)
        figure_names.append(name)

    fig, ax, foot = axes("各地形成功率随训练时间变化", "成功率", time_axis=True)
    for f, family in enumerate(families):
        ax.plot(x, [s["success_rate"][f] for s in stats], color=COLORS[f], lw=2.3, label=LABELS[family])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, ax, 1, "success_rate_over_time", foot)

    fig, ax, foot = axes("各地形平均回合长度随训练时间变化（估算）", "平均回合长度（秒，估算）", True, True)
    for f, family in enumerate(families):
        ax.plot(x, [s["episode_steps_estimate"][f] * dt for s in stats], color=COLORS[f], lw=2.3, label=LABELS[family])
    ceiling = max(10., max(float(np.nanmax(s["episode_steps_estimate"])) * dt for s in stats) * 1.04)
    ax.set_ylim(0, ceiling)
    secondary = ax.secondary_yaxis("right", functions=(lambda s: s / dt, lambda steps: steps * dt))
    secondary.set_ylabel("控制步（50 Hz）", fontsize=11)
    save(fig, ax, 2, "episode_length_over_time_estimated", foot)

    fig, ax, _ = axes("各地形平均课程难度随训练时间变化", "平均课程等级（L0–L9）", True)
    for f, family in enumerate(families):
        ax.plot(x, [s["mean_level"][f] for s in stats], color=COLORS[f], lw=2.3, label=LABELS[family])
    ax.set_ylim(0, 9)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    save(fig, ax, 3, "mean_curriculum_level_over_time",
         "直接读取 Curriculum/<terrain>/mean_level，再对窗口内各轮快照取均值。\n平地的等级变化仅表示课程分配变化；flat 在 L0–L9 的几何均为平地。")

    def mark_low_samples(ax, metric):
        # Open markers distinguish sparse cells from well sampled levels.
        for f, family in enumerate(families):
            selected = (final["ended"][f] > 0) & (final["ended"][f] < 100)
            for level in np.flatnonzero(selected):
                y = metric[f, level]
                ax.plot(level, y, "o", ms=7, mfc="white", mec=COLORS[f], mew=1.8, zorder=5)
                dy = 14 if family == "slope" else (17 if level == 8 else -24)
                ax.annotate(f"n={final['ended'][f, level]}", (level, y), xytext=(-6, dy),
                            textcoords="offset points", fontsize=10, color=COLORS[f], ha="right")
    sparse = [f"{family} L{l}={int(final['ended'][f,l])}" for f, family in enumerate(families)
              for l in range(10) if 0 < final["ended"][f,l] < 100]
    empty = [f"{family} L{l}" for f, family in enumerate(families)
             for l in range(10) if final["ended"][f,l] == 0]
    sparse_note = "空心点：结束回合少于 100（" + "、".join(sparse) + "）。"
    if empty:
        sparse_note += "无样本留空：" + "、".join(empty) + "。"

    fig, ax, foot = axes(f"各地形、各难度的成功率 · 最后 {last:,} 轮", "成功率")
    for f, family in enumerate(families):
        ax.plot(range(10), final["success_rate"][f], color=COLORS[f], lw=2.3, marker="o", ms=5, label=LABELS[family])
    mark_low_samples(ax, final["success_rate"])
    ax.set_ylim(-.025, 1.025)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, ax, 4, "last1000_success_rate_by_level", foot + "\n" + sparse_note)

    fig, ax, foot = axes(f"各地形、各难度的平均回合长度 · 最后 {last:,} 轮（估算）", "平均回合长度（秒，估算）", estimate=True)
    for f, family in enumerate(families):
        ax.plot(range(10), final["episode_steps_estimate"][f] * dt, color=COLORS[f], lw=2.3, marker="o", ms=5, label=LABELS[family])
    mark_low_samples(ax, final["episode_steps_estimate"] * dt)
    ax.set_ylim(0, max(10., float(np.nanmax(final["episode_steps_estimate"])) * dt * 1.04))
    secondary = ax.secondary_yaxis("right", functions=(lambda s: s / dt, lambda steps: steps * dt))
    secondary.set_ylabel("控制步（50 Hz）", fontsize=11)
    save(fig, ax, 5, "last1000_episode_length_by_level_estimated",
         "估算：环境控制步暴露量 ÷ 已结束回合数；存在窗口边界和轮内等级迁移误差，详见计算说明。\n" + sparse_note)

    def rows_for(summary, start, stop, by_level=False):
        for f, family in enumerate(families):
            for level in (range(10) if by_level else [None]):
                ix = (f, level) if by_level else f
                row = dict(terrain=family, level=level if level is not None else "all",
                    iteration_start=start, iteration_end=stop-1,
                    elapsed_hours_start=float(data["elapsed_hours"][start]),
                    elapsed_hours_end=float(data["elapsed_hours"][stop-1]),
                    completed_episodes=int(summary["ended"][ix]),
                    success_rate=float(summary["success_rate"][ix]),
                    course_success_rate=float(summary["course_success_rate"][ix]),
                    mean_episode_steps_estimated=float(summary["episode_steps_estimate"][ix]),
                    mean_episode_seconds_estimated=float(summary["episode_steps_estimate"][ix] * dt),
                    mean_episode_seconds_bound_low=float(summary["episode_steps_bound_low"][ix] * dt),
                    mean_episode_seconds_bound_high=float(summary["episode_steps_bound_high"][ix] * dt),
                    exposure_control_steps=float(summary["exposure"][ix]),
                    exposure_integration_error_bound=float(summary["exposure_error_bound"][ix]))
                row.update({outcome: int(summary["counts"][ix][i]) for i, outcome in enumerate(OUTCOMES)})
                row["mean_curriculum_level"] = float(summary["mean_level"][f])
                yield row
    series_rows = [r for (a, b), summary in zip(blocks, stats) for r in rows_for(summary, a, b)]
    last_rows = list(rows_for(final, n-last, n, True))
    last_family_rows = list(rows_for(final_family, n-last, n))
    write_csv(output / "time_series_200_iterations.csv", series_rows)
    write_csv(output / "last1000_by_terrain_level.csv", last_rows)
    write_csv(output / "last1000_by_terrain.csv", last_family_rows)
    write_csv(output / "global_recent100_episode_length_tensorboard.csv", (
        dict(iteration=i, elapsed_hours=float(data["elapsed_hours"][i]),
             mean_episode_steps=float(data["global_recent100_episode_steps"][i])) for i in range(n)))
    summary = dict(final1000_by_terrain=last_family_rows, final1000_by_terrain_level=last_rows,
        first_window=series_rows[:5], last_window=series_rows[-5:],
        final_mean_levels=data["mean_level"][-1].tolist(),
        final1000_env_snapshot_mean_levels=data["mean_level"][-last:].mean(0).tolist(),
        figures=[f"{name}.png" for name in figure_names])
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    (output / "summary.json").write_text(json.dumps(finite_json(summary), ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n")
    write_report_text(meta, summary, output)
    print(json.dumps({"final_mean_levels": summary["final_mean_levels"],
                      "final1000_by_terrain": last_family_rows}, ensure_ascii=False, indent=2), flush=True)


def write_report_text(meta, summary, output):
    # Detailed methods are kept with the exported numerical values and figures.
    final = summary["final1000_by_terrain"]
    table = ["| 地形 | 已结束回合 | 成功率（全部结束回合） | 课程成功率 S/(S+F) | 平均回合长度估算 s | 平均等级 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in final:
        table.append(f"| {r['terrain']} | {r['completed_episodes']:,} | {r['success_rate']:.2%} | "
            f"{r['course_success_rate']:.2%} | {r['mean_episode_seconds_estimated']:.3f} | {r['mean_curriculum_level']:.3f} |")
    rows = ["# PGMT Stage2 训练地形指标汇总", "",
        f"来源：`{meta['checkpoint']}`。连续 {meta['iterations']:,} 轮、{meta['num_envs']:,} 环境，"
        f"每轮 {meta['steps_per_iteration']} 控制步，控制周期 {meta['dt']} s。实际训练时间约 {meta['training_hours']:.2f} 小时。",
        "", "本报告只读取已有数据，没有重新评测或修改模型。成功率和难度来自真实日志；两张回合长度图是已获用户确认的估算。",
        "", "## 最后 1,000 轮汇总", "", *table, "",
        "窗口为零基 iteration 19000–19999，等价于第 19001–20000 次已完成 PPO 更新。"
        "这 1,000 轮使用了不断更新的策略，不能解释为 model_19999.pt 的固定策略独立测试结果。", "",
        "## 主要走势", "",
        "- 楼梯从最初 200 轮约 24.74% 成功率升到最后 1000 轮 49.58%，平均课程等级升至 3.24；箱体从约 24.62% 升至 41.50%，但平均等级仅约 1.12。",
        "- 平地、rough 的成功率和平均等级在后半程下降；flat 的几何不随等级变化，不能把其下降直接归因为地形变难。动作级自适应采样使动作分布变化，当前日志不足以把策略退化与抽样分布变化分开。",
        "- 高等级楼梯、斜坡、箱体的成功率及长度估算下降；rough 下降较缓，flat 在各等级约 57%。这是训练分布下的观察，未控制动作组成。",
        "- 最后 1000 轮 boxes L9 无结束回合，L8 仅 2 个、L7 仅 16 个；slope L9 仅 19 个。图中用空心点标记少于 100 个的格子，无样本格子留空，CSV 保留全部样本数。",
        "", "## 五张折线图", ""]
    for name in summary["figures"]:
        rows += [f"![{name}]({name})", ""]
    rows += ["## 数据来源、计算方法与限制", "",
        "### 1. 分组及成功率", "",
        "从 progress.jsonl 每轮读取 `OutcomeWindow/{terrain}/L{level}/{outcome}`。"
        "outcome 包含 success、failure、map_exit、timeout、incomplete_motion、other_truncation。"
        "按回合出生时的地形/等级归组，跨 tile 后不重新归组；与课程结算一致。"
        "success 使用课程完整跟踪判据：从第 0 帧开始、到达末帧且实际步数足够，并未被失败/越界等更高优先级事件覆盖；不是 motion sampler 的 success。", "",
        "令窗口内成功数为 S、失败数为 F、四类中性截断总数为 Z：", "",
        "- 图中成功率 = S / (S + F + Z)，沿用分地形日志的 `success_fraction_of_outcomes` 定义。",
        "- 同时导出的课程成功率 = S / (S + F)，与 `Curriculum/terrain_progress/success_rate` 定义一致。",
        "- 多轮、多等级聚合时先加总分子与分母，再求比值；不对逐轮成功率做无权平均。",
        "- 空分母记为 NaN，不把没有样本的格子当作 0% 成功。", "",
        "### 2. 平均课程难度", "",
        "每轮直接读取 `Curriculum/{terrain}/mean_level`，并与 `sum(level × env_count) / sum(env_count)` 复核。"
        "图 3 对每 200 轮的快照做算术平均。它表示 L0–L9 的课程等级，不是高度、坡度，也不是把等级除以 9 得到的实际地形 difficulty。"
        "flat 十级几何相同，因此其等级上升只代表课程分配上升。", "",
        "### 3. 平均回合长度为何只能估算", "",
        "日志没有按出生地形/等级保存 `episode_length_sum` 或逐回合 terminal length。"
        "TensorBoard 仅有全体环境最近 100 个已结束回合的 `Train/mean_episode_length`，单位为控制步；"
        "该滑动窗口没有地形标签，也不是一轮内全部已结束回合的均值，无法据此反推出分地形长度。"
        "checkpoint 也只保存课程结果计数与当前等级，没有终止回合长度。", "",
        "采用窗口暴露量估算：`估算平均步数 L_hat = E_hat / D`，`估算秒数 = L_hat × 0.02`。"
        "D 是窗口内全部已结束回合数，包含成功、失败和中性截断。", "",
        "**按地形：** 各地形的环境数量 N_f 在整个训练中固定，窗口有 W 轮，因此"
        "`E_f = 24 × W × N_f`。这个累计环境控制步数可以精确得到，但比值仍不是完成回合长度的精确均值。", "",
        "**按地形及等级：** 令 n_i 为第 i 轮结束时该格子的 `env_count`，则"
        "`E_hat_(f,l) = sum_i [24 × (n_(i-1) + n_i) / 2]`。"
        "课程可能在一轮内部升降级，日志没有逐控制步人数，故用相邻快照作梯形积分。"
        "最后 1000 轮的首个边界使用 iteration 18999 的快照。首轮初始人数则由首轮末人数减去已记录的升降级净流量恢复。", "",
        "**误差来源与可核验边界：** 令 A_start、A_end 为窗口两端仍在进行的回合已累计年龄之和，单位为控制步。"
        "真实已结束回合总长度满足 `sum(length_completed) = E_true + A_start - A_end`。"
        "现有日志没有这两个年龄和，因此窗口跨界回合会带来偏差；不能把暴露量比值声称为精确均值。", "",
        "一次等级迁移对 24 步快照梯形积分的误差至多 12 个环境控制步。"
        "用成功/失败导致的流入和流出次数 M，可得 `B_E = 12 × M`；边界 L0 失败、L9 成功不会迁移，不计入 M。"
        "按地形汇总时迁移不改变地形归属，故 `B_E=0`。", "",
        "设 B=500 为最大回合步数，窗口起末人数分别为 N_start、N_end。真实已结束回合均值的保守可行边界为：", "",
        "```text",
        "lower = max(0, (E_hat - B_E - B × N_end) / D)",
        "upper = min(B, (E_hat + B_E + B × N_start) / D)",
        "```", "",
        "首个训练窗口的 A_start=0，可去掉上界中的 B×N_start。边界已乘 dt 后写入 CSV 的 "
        "`mean_episode_seconds_bound_low/high`；这是确定性保守边界，不是统计置信区间。"
        "估算值不被强行裁剪到 10 秒以内；本次图中所有有限值均保留计算结果。", "",
        "### 4. 时间轴与平滑", "",
        "前三张图每 200 轮形成一个互不重叠的窗口，点位取该窗口首末记录墙钟时间的中点；不额外做 EMA。"
        "下轴为 TensorBoard `Perf/total_fps` 对应的 wall_time，减去首轮该时间戳再加回首轮 collection+update 用时，换算为小时；"
        "因此包含训练期间的日志/保存开销，不包含仿真器启动。起点有首轮记录调用前少量开销的不确定性。"
        "上轴为根据实际 wall_time 对齐的已完成 PPO 轮数。", "",
        "最后两张图以 L0–L9 为横轴，在最后 1000 轮内先合并计数和暴露量，绘制每个地形的一条线。"
        "它们不是 50 条时间曲线。各等级的动作抽样分布、课程占用和结束样本量不同，"
        "曲线不一定随难度单调；不能当作相同动作集、均衡采样下的因果难度比较。", "",
        "### 5. 数据核验与导出", "",
        "已逐轮核验：iteration 连续；OutcomeWindow 等于累计计数差；结局总数等于 done 审计；"
        "最后累计计数与指定 checkpoint 完全一致；地形人数恒定；等级人数变化与课程升降级流量完全一致；"
        "mean_level 与人数加权等级一致。TensorBoard 时间轴覆盖全部 20000 轮且严格递增；读取的目标事件做 CRC32C 校验。", "",
        "- `time_series_200_iterations.csv`：三张时间曲线的绘图数据。",
        "- `last1000_by_terrain_level.csv`：50 个地形×等级的计数、两种成功率、长度估算、误差边界。",
        "- `last1000_by_terrain.csv`：最后 1000 轮按地形汇总。",
        "- `global_recent100_episode_length_tensorboard.csv`：全局原始滑动窗口长度，仅作参考，不用于分地形估算。",
        "- `metrics_arrays.npz`：20000 轮原始分组计数、人数、等级与时间戳及暴露量数组。",
        "- `metadata.json`：输入哈希、统计参数和核验结果；`summary.json`：汇总数字。",
        "- 同名 SVG：五张图的矢量版本。", "",
        "复现命令：", "", "```bash", "cd /home/lzp/projects/perceptive_tracking/PMT",
        "MPLCONFIGDIR=/tmp/pgmt-analysis-mpl /home/lzp/miniconda3/envs/pmt/bin/python \\",
        "  scripts/analyze_pgmt_training_metrics.py \\",
        f"  --checkpoint {meta['checkpoint']} \\",
        f"  --output {output}", "```", ""]
    (output / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--last", type=int, default=1000)
    args = parser.parse_args()
    run, output = args.checkpoint.resolve().parent, args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, metadata = read_progress(run, args.checkpoint)
    wall, global_length, event_info = read_event_clocks(run, metadata["iterations"])
    metadata["tensorboard"] = event_info
    # First rollout starts just before its measured collection + learning work.
    origin = wall[0] - data["compute_times"][0].sum()
    elapsed_h = (wall - origin) / 3600
    data.update(wall_time=wall, elapsed_hours=elapsed_h, global_recent100_episode_steps=global_length)
    metadata.update(checkpoint=str(args.checkpoint.resolve()), run=str(run),
        analysis_script_sha256=digest(Path(__file__)),
        generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        plot_window_iterations=args.window, final_window_iterations=args.last,
        wall_origin_epoch=origin, wall_origin_method="first Perf/total_fps wall_time minus first collection_s and update_s",
        training_hours=float(elapsed_h[-1]),
        final_window_start_iteration=metadata["iterations"]-args.last,
        final_window_end_iteration=metadata["iterations"]-1)
    np.savez_compressed(output / "metrics_arrays.npz", **data)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Extracted {metadata['iterations']} iterations, {elapsed_h[-1]:.3f} hours", flush=True)
    create_report(data, metadata, output, args.window, args.last)


if __name__ == "__main__":
    main()
