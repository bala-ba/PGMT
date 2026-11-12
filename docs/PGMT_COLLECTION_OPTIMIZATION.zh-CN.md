# PGMT collection 优化实施与验收（2026-10-15）

这是 [collection 分析记录](PGMT_COLLECTION_PERFORMANCE.zh-CN.md) 中候选优化的实施记录。测试使用 `pmt` / Isaac Sim 5.1.0、CUDA 0、8192 环境、headless、完整 v6 地形。正式训练停留在 383 次更新，其 checkpoint 不被测试覆盖。

最终两组共享 GPU 复测中，collection 从 **14.564 s 降到 10.373 s（−28.8%，吞吐 ×1.40）**；collection + PPO 从 **18.354 s 降到 14.225 s（−22.5%）**。首轮 all 组合曾测到 4.701 s，但当时 GPU 0 基本空闲，不能与其他组直接计算受控加速比。

## 比较方法

- 每组从同一 `formal_383_numpy1.pt`、seed 722 恢复，进行 8 次 PPO 更新；每轮 24 控制步、96 物理子步，比较后 6 轮，前 2 轮用于热身。
- 前六组逐项累计开启开关，另补关闭审计、最终原版和完整优化复测。增量收益依赖已启用的其他开关，不能将表中百分比直接相加。
- 所有组使用相同轻量 wall timers，不启用原生 PhysX profiler。初始模型、机器人状态、任务配置、算法配置均独立比对。
- 物理步长、solver、动作兼容表、随机放置规则、奖励权重、终止条件、PPO rollout 长度和更新设置不变。当前仍固定源动作第 0 帧，XY/yaw 随机化保留。
- 同机存在其他 GPU 任务，wall time 包含异步 CUDA 等待；单个嵌套计时不能视为对应 GPU kernel 的独立耗时。结果是短训测量，不代表长期平均值。

## 已实施的开关

| 开关 | 改动 | 保留的语义 |
|---|---|---|
| `reset_stats` | command/curriculum/termination 统计留在 GPU；sampler 指标在 EMA/恢复时缓存；motion outcome 使用 scatter 累加 | reset 日志拥有旧 episode 的值，失败/成功各计一次；动作采样概率与兼容过滤不改 |
| `placement` | 缓存固定第 0 帧源数据和未来位置；碰撞放置只变换所需几何；未来越界只变换位置；reset 时局部更新参考/指标；使用 `index_fill_` 清理缓冲区 | 每次仍重采样 XY/yaw、查询实际地形、检查碰撞/边界并按原顺序重试；保留 terminal 奖励快照 |
| `contact_sensor` | 全身 `contact_forces` 的 history 从 3 改为 0、关闭原生 air-time；由读取当前力触发更新 | 当前全对象力、两只脚的地形过滤传感器、PGMT 自有接触事件/时间状态不改 |
| `audit` | 每步在 GPU 检查全部 reward 样本，按 PPO 周期统一读取结果 | 仍检查 finite、四头之和、逐头加权分项，并在 returns/GAE/PPO 前阻止错误 rollout；日志不降采样 |
| `reward_compile` | 使用 `torch.compile(fullgraph=True, dynamic=False)` 编译 21 项纯张量奖励数学核 | 六项 terrain-contact、物理推进、reset 和 terminal 缓存仍使用原实现；权重和 dt 仍只应用一次 |

原始候选中的“改写分组动作采样”没有实施：细分测量 `_draw_samples` 约 0.061 s/轮，收益较小。保留分组 `multinomial` 和 RNG 顺序，只优化 outcome 累加及指标日志。也没有缓存 adaptive 概率或绕过地形校验。

代码入口：

- [开关、统计、传感器与审计](../pmt_tasks/pgmt_performance.py)
- [放置缓存、局部更新及 reset 清零](../pmt_tasks/pgmt_fast_placement.py)
- [command 放置分支](../pmt_tasks/mdp/commands/pgmt_motion_command.py)
- [几何专用变换](../pmt_tasks/pgmt_reference.py)、[奖励/接触缓存生命周期](../pmt_tasks/pgmt_runtime.py)
- [训练开关和 GAE 前审计](../scripts/train_pgmt_stage2.py)、[checkpoint 执行选项](../motion_tracking_rl/runners/pgmt_runner.py)

## 测量结果

每组均为 8 次 PPO，表中为后 6 次的平均值；± 为样本标准差。前六行按顺序累计开启开关；后三行用于额外对照。

| 测试目录 / 开关 | Collection（s/轮） | PPO（s/轮） | Reset（s/轮） | 启动前 GPU 0 |
|---|---:|---:|---:|---|
| `baseline_v2` / none | 14.497 ± 0.381 | 3.827 | 6.503 | 89%，共享 |
| `reset_stats` | 14.037 ± 0.602 | 3.852 | 6.341 | 93%，共享 |
| 上项 + `placement` | 11.284 ± 0.228 | 3.864 | 3.764 | 94%，共享 |
| 上项 + `contact_sensor` | 10.461 ± 0.639 | 3.796 | 3.606 | 94%，共享 |
| 上项 + `audit` | 11.647 ± 0.148 | 4.008 | 4.016 | 94%，共享 |
| 上项 + `reward_compile`，即 all | 4.701 ± 0.064 | 2.439 | 1.334 | **0%，低负载，不可直接归因** |
| `optimized_no_audit` / all 去掉 audit | 10.609 ± 0.244 | 3.849 | 3.605 | 87%，共享 |
| `baseline_final` / none 复测 | **14.564 ± 0.466** | 3.790 | 6.543 | 88%，共享 |
| `optimized_repeat` / all 复测 | **10.373 ± 0.436** | 3.852 | 3.582 | 86%，共享 |

逐项结论：

1. `reset_stats` 首次观察到约 3.2% 的 collection 改善，幅度较小，不能认定全部 reset 开销来自 `.item()`。
2. `placement` 累计组比前一组约快 19.6%，reset 明显下降。其中 RewardManager 的 reset 从 0.857 s/轮降至 0.010 s/轮，GPU `index_fill_` 清零是明确的有效改动。没有删除几何校验或物理 `forward`。
3. `contact_sensor` 累计组比前一组约快 7.3%；全身 sensor 更新从 **96 次降为 24 次/轮**，该调用段从 0.481 s 降到 0.109 s。未更改脚的地形过滤传感器。
4. `audit` 降低了包装器自身耗时，但首次加入后 collection 反而增大，独立净收益未证实。最终 all 相比去掉 audit 仅快约 2.2%，小于组间波动所允许的归因精度；不据此宣称其有稳定单项提速。
5. `reward_compile` 已通过数值、训练与完整组合复测。首次 4.701 s 的大幅下降混入了 GPU 负载改变；**不能声称编译单项获得约 60% 提速，也不能据此证明 audit/compile 的巨大交互收益**。它的独立增益仍需受控独占 GPU 实验才能量化。

两次 none 基线接近（14.497 / 14.564 s）。最终 all 的 collection 吞吐从 13,500 增至 18,954 transitions/s；reset 从 6.543 降至 3.582 s；物理推进从 3.479 到 3.531 s，基本没有改善，因此本次可靠收益主要在物理推进之外。两次最终比较仍不是独占 GPU 实验，背景作业负载未受控。

首次 all 的首轮 collection 为 11.991 s，包含编译准备，不计入上述后 6 轮均值。编译可能在新环境或输入形状变化后重新准备，收益应看热身后的多轮表现。8192 环境全部短训均无 OOM；完整组合首次测得 Torch 峰值 allocated 约 15,274 MiB、reserved 16,598 MiB，均不包含全部 Isaac/驱动和其他进程显存。

机器可读数据保存在 [summary.json](../logs/pgmt_speed_20261015/summary.json)，GPU 背景快照在 [gpu_load_context.json](../logs/pgmt_speed_20261015/gpu_load_context.json)。所有通过组都包含训练状态、逐轮计时、最终 checkpoint、27 项奖励及四头 loss 的 TensorBoard 数据。原始目录中也保留失败尝试，未将其纳入通过组统计。

## 正确性检查

- 相同起点：各组初始模型及物理状态逐项完全相同，任务/算法配置相同。
- 放置：五类地形 × L0/L5/L9 × 8 环境，共 120 次，优化/原实现使用相同 RNG；平移、旋转、落点、patch ID、最小间隙、重试次数和 RNG 消耗完全一致。
- 接触：全身 sensor 与直接 PhysX 读数的 `norm > 10 N` 分类完全一致；原接口连续读取本身存在约 `1.22e-4 N` 的浮点变化，因此数值检查使用 `rtol=3e-5, atol=1e-4`，并额外要求分类零差异。
- 编译奖励：对实际 8192 环境逐项比较 21 项 eager/compiled 输出，要求 `rtol=2e-5, atol=5e-5`。最终复测的最大误差仅占该容差上限的 0.0137；乘实际权重与 dt 后，最大单项每步奖励差为 **2.5e-8**。原始力平方等高量纲项的绝对误差不能直接当作每步奖励误差。详见 [optimization_checks.json](../logs/pgmt_speed_20261015/optimized_repeat/optimization_checks.json)。
- 环境链路：保留参考变换、地形高度/掩膜、真实 PhysX 状态、受控 reset 和额外 8 步接触/参考更新检查。
- 训练链路：所有四个 value head、selector、patch encoder、IFM、actor/critic 均有有限梯度和参数更新；模型和 optimizer tensors 有限；课程新增 outcome 总数严格等于 done 总数，初始恢复不制造虚假 outcome。
- 日志：27 项 `RewardStep/*` 均完整记录 8 个更新，全部 scalar 有限；每轮审计仍覆盖 24 控制步。
- CPU 回归：`pmt` 核心测试 62 passed / 2 CUDA skipped；历史 checkpoint/四头测试在其 NumPy 2 保存环境中 39 passed / 3 CUDA skipped。CUDA 正确性另外由真实 Isaac 短训和数值对照验证。

原生入口另做严格续训：从 `optimized_repeat/final.pt` 的 next_iteration=391 恢复，追加 2 次 PPO 到 **393**。独立 CPU 复核初始模型、optimizer、环境状态、学习率、训练契约及执行选项均与来源完全相同；141 个 optimizer tensors 有限；40 次新增课程 outcome 与 40 次 done 一致。TensorBoard 包含 1078 个 scalar tags，27 项奖励均在 iteration 391/392 连续记录且有限。两轮 collection 为 10.251 / 9.735 s；这两轮只用于续训验收，不混入上述 8 轮 benchmark 均值。见 [strict_resume_report.json](../logs/pgmt_speed_20261015/strict_resume_report.json)。

验收结束后全部测试 worker 已退出，GPU 上不再有本次测试进程。正式训练仍停留在 383，`logs/pgmt_p4_20261014/formal_8192/final.pt` 的 SHA256 仍为 `a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`；测试到 391/393 的文件均位于独立目录。见 [进程与正式 checkpoint 核对](../logs/pgmt_speed_20261015/final_process_audit.json)。

曾出现的失败均保留记录：首次诊断因 Python 包名遮蔽在 PPO 前失败，改用 `importlib.import_module` 后重跑；首次 contact 检查要求力逐位一致而失败，加入原始接口重复读取对照后调整为数值容差及严格分类一致，重新完成 8 轮；旧 NumPy 2 checkpoint 在 NumPy 1 中不能直接加载，训练使用已逐项验证等价的 NumPy 1 副本，未修改安装环境。

## 使用与恢复

以下 383 更新权重与命令属于 2026-10-15 性能实验的历史配置。当前版本增加了 timeout 边界、中央 2×2 出生和新 T1 契约，不能直接用这些旧权重严格续训；当前命令见 [使用指南](USER_READ.md#正式四头地形训练与续训)。性能实验记录保留原始口径。

训练 Python 入口默认 `--performance none`，原路径可随时恢复。新增 [原生 pmt 启动脚本](../scripts/train_pgmt_stage2_native.sh) 默认 `--performance all`、headless、CUDA 0、四头 terrain task，末尾参数可以覆盖这些选项。原 Sim 6 GUI 启动脚本保留。

以下命令只执行 8 次更新；使用新的日志目录名：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash scripts/train_pgmt_stage2_native.sh \
  --num_envs 8192 --max_iterations 8 --save_interval 8 \
  --resume logs/pgmt_collection_profile/formal_383_numpy1.pt \
  --log_dir logs/pgmt_speed_recheck
```

关闭全部优化用 `--performance none`；选择部分优化用逗号分隔，例如 `--performance reset_stats,placement,contact_sensor`。后者是无需编译、且主要收益已有证据支持的配置；`all` 增加通过数值验收的审计/编译选项，不保证每个开关都带来独立提速。

开关写入 `performance.json`、`training_status.json` 和 checkpoint 的 `execution_options`，与训练语义契约分开保存。严格恢复仍检查模型、normalizer、optimizer、课程、sampler 概率和更新编号；恢复时执行开关由命令指定，不自动套用旧 checkpoint 的开关。

完整 benchmark 的复现入口在 `logs/pgmt_speed_20261015/run.sh`，带细分计时与额外功能后检；上面的原生训练入口不附加诊断包装器。

查看本次全部测试曲线：

```bash
/home/lzp/miniconda3/envs/pmt/bin/tensorboard \
  --logdir /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_speed_20261015 \
  --host 127.0.0.1 --port 6007 --load_fast=false
```

完整过程：[EXECUTION.md](../logs/pgmt_speed_20261015/EXECUTION.md)。现有 6006 TensorBoard 仍服务原正式训练目录。

## CUDA 1 训练效率实测

2026-10-15 按用户要求切换到物理 GPU 1，保持 `pmt` / Isaac Sim 5.1、8192 环境、headless、`--performance all`、seed 722、24 步 rollout 和全部训练设置。完成 8 次 PPO，仍排除前 2 轮，统计后 6 轮。对照为前面的 CUDA 0 `optimized_repeat`。

| 指标 | CUDA 0 | CUDA 1 | 变化 |
|---|---:|---:|---:|
| Collection / 轮 | 10.373 ± 0.436 s | **5.002 ± 0.117 s** | 耗时 −51.8%，吞吐 ×2.07 |
| Returns + PPO / 轮 | 3.852 s | **2.878 s** | 耗时 −25.3% |
| Collection + PPO / 轮 | 14.225 s | **7.879 s** | 耗时 −44.6% |
| Collection 吞吐 | 18,954 transitions/s | **39,309 transitions/s** | +107.4% |
| Reset / 轮 | 3.582 s | 1.380 s | 主机 wall time，含 CUDA 等待 |
| 96 次物理推进 / 轮 | 3.531 s | 2.110 s | 同上 |
| RewardManager / 轮 | 1.446 s | 0.854 s | 同上 |

结论：**当前机器负载下，后续训练优先选择 CUDA 1 有明确实测依据。** 两张卡均为 RTX 5090；CUDA 0 对照启动前已有约 86% 的后台负载，CUDA 1 本次启动前约 14%。这是共享机器在不同后台负载下的顺序测试，不能解释为 GPU 1 硬件本身快一倍。CUDA 1 仍承担桌面显示并有另一计算任务。首轮 collection=11.770 s 包含编译准备，未计入表中后 6 轮均值。

设备与恢复：

- AppLauncher 的 `active_gpu=1 / physics_gpu=1`；实际 PhysX `device_ordinal=1`。策略参数、机器人 tensor、完整动作存储和 Warp 查询网格均为 `cuda:1`，见 [设备核验](../logs/pgmt_speed_20261015/cuda1_all_8192/device_evidence.json)。
- 原 checkpoint 的严格训练契约固定了 `sim.device` 与 `commands.motion.storage_device`。测试使用独立副本 `logs/pgmt_speed_20261015/formal_383_cuda1.pt`，仅将这两个字段从 `cuda:0` 改为 `cuda:1`；211 个 tensors、1 个 ndarray 和其余标量逐项完全相同，模型/optimizer/课程/sampler/RNG 不变。未放宽严格恢复检查，原 checkpoint 未改动。见 [设备迁移核验](../logs/pgmt_speed_20261015/cuda1_checkpoint_device_port.json)。
- 与 CUDA 0 对照相比，初始模型和所有记录的初始物理状态完全相同；配置差异仅为设备编号（包括 agent 内嵌的契约副本）。

验收与显存：

- 8 次 PPO 从 383 更新至 391；27 项奖励、1078 个 TensorBoard scalar tags 均记录完整且有限，四个 value head 均有有限非零梯度，141 个 optimizer tensors 有限。
- 192 个控制步出现 1576 次 done，新增课程 outcome 也为 1576，无重复结算。地形/扫描、参考变换、接触力分类、120 次放置对照及 compiled/eager 奖励数值检查全部通过。
- 8192 环境无 OOM。Torch 峰值 allocated=15,275.8 MiB、reserved=16,598 MiB。运行期间一次快照：本训练进程在 GPU 1 占 25,038 MiB；整卡已用 29,481 MiB、剩余 2,628 MiB。该快照包含桌面和其他作业，不是连续采样峰值。Isaac 另在 GPU 0 保留约 532 MiB 初始化上下文，实际训练设备核验为 GPU 1。
- worker PID 1280445 已正常退出，GPU 分配已释放。正式 383 checkpoint SHA256 不变，未启动长期训练。

完整结果：[cuda1_report.json](../logs/pgmt_speed_20261015/cuda1_report.json)；原始日志与 TensorBoard：`logs/pgmt_speed_20261015/cuda1_all_8192/`；[显存快照](../logs/pgmt_speed_20261015/cuda1_memory_snapshot.json)、[退出核验](../logs/pgmt_speed_20261015/cuda1_process_exit.json)。使用新的日志目录名复现同样的有限测试：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash logs/pgmt_speed_20261015/run_cuda1.sh cuda1_all_8192_repeat
```

## 后续：actor/critic 同批共享编码

2026-10-15 已完成联合前向，正常 PGMT PPO 自动启用，详见 [实现与验收记录](PGMT_SHARED_ENCODING.zh-CN.md)。同一 CUDA 1 / 8192 环境的顺序 A/B 短测中，PPO 从 2.856 降至 1.507 s/轮（−47.2%），collection + PPO 从 7.950 降至 6.556 s/轮（−17.5%），Torch 峰值 allocated 从 15,274.4 降至 11,766.0 MiB。每轮训练编码次数从 89 降至 45；collection 本身变化较小。

没有新增模型参数、观测增强或跨 minibatch 缓存。旧 checkpoint 严格恢复检查保留，执行方式单独记录为 `execution_options.actor_critic_forward=joint_encode`；正常入口 391→393 续训已通过。以上旧 collection 开关不控制此项联合前向；`--performance none` 仅关闭原表中的环境执行优化。

## 后续：PPO 日志标量集中读取

2026-10-15 已默认接入 GPU 日志累计与轮末一次读取，保留 KL 自适应学习率和即时异常检查。当前四头配置每轮常规日志读取从 181 次减至 1 次，日志/梯度/参数/Adam/RNG 与旧路径对照通过，正常入口严格续训通过。

CUDA 1 / 8192 环境短测：PPO 1.522→1.506 s，collection + PPO 6.576→6.563 s。同步次数的减少已确认，但本次约 0.20% 的整轮变化不足以认定稳定提速。详见 [实现与验收记录](PGMT_PPO_LOG_SCALARS.zh-CN.md)。此项同样独立于 `--performance` 环境执行开关。
