# PGMT collection 耗时诊断

2026-10-15。正式训练按用户要求停在 **383 次 PPO 更新**，`logs/pgmt_p4_20261014/formal_8192/final.pt` 保持不变。本次只做独立有限诊断，未修改生产训练源码、物理设置、奖励或接触判据。诊断进程均已正常退出。

## 结论

**直接瓶颈是原生物理推进，单次 physics substep 约 0.824 秒。** 每轮 24 个控制步、每控制步 4 个物理子步，共 96 次推进，因此仅物理推进就耗时约 79.15 秒。该轮 collection 为 80.63 秒，物理推进占 98.16%。

正式训练最后检查的 50 轮平均 collection 99.30 秒、PPO 2.43 秒；本次从 checkpoint 恢复后测量首个 24 步 rollout，collection 较短，不能把它当作稳态加速结果。本次没有 reset，也没有 render，说明不依赖 reset 就能复现严重的采样耗时。

## 测量方法与结果

使用相同 CUDA 0、8192 环境、v6 地形、四头策略，从正式 `final.pt` 严格恢复；两次诊断各仅执行 1 个完整更新，结果位于独立目录，诊断 checkpoint 的 next_iteration=384 不替代正式 checkpoint。

第一次 cProfile 输出累计仅 9.33 秒，但原生 runner 的 collection 为 87.22 秒，调用次数与 96 次物理推进也不完整，故不能用该报告分摊实际耗时。第二次对各 Python 调用边界使用 `time.perf_counter()` 直接计时，记录调用次数、包含子调用的耗时和扣除已包装子调用后的耗时；不依赖 Python profiler hook，不额外插入 CUDA synchronize。

| 环节 | 调用次数 | 墙钟时间 |
| --- | ---: | ---: |
| Collection 总计 | 1 | 80.6308 s |
| `sim.step(render=False)` | 96 | 79.1463 s |
| 其中 `PhysicsContext._step` | 96 | 79.1356 s |
| RewardManager 计算全部奖励 | 24 | 0.7981 s |
| ObservationManager | 25，含 learn 开始读观测 | 0.1513 s |
| TerminationManager | 24 | 0.1213 s |
| 策略采样 `alg.act` | 24 | 0.1010 s |
| 步级包装/奖励审计自身，扣除环境和 telemetry | 24 | 0.0313 s |
| 双脚 terrain sensor 读取合计 | 48 | 0.0066 s |
| Returns + PPO 更新，位于 collection 之外 | 1 | 2.5010 s |

表中父/子环节存在包含关系，不能全部相加。传感器和其他 GPU 操作可能异步执行，等待时间会记入实际阻塞的外围调用；这些是主机墙钟时间，不是各 GPU kernel 的独占执行时间。原生 `simulate` 内部的碰撞、求解、回调和 CPU/GPU 同步尚未细分。

## 代码与排除项

- Rollout 边界：[on_policy_runner.py](../motion_tracking_rl/runners/on_policy_runner.py)，`learn()` 中 collection 包含 24 次动作推理、环境推进和步级记录。
- 控制步推进：本地 Isaac Lab `isaaclab/envs/manager_based_rl_env.py:182–189`，`decimation=4`，每步调用四次 `self.sim.step(render=False)`。
- 最深已计时边界：本地 Isaac Sim 6 `extsDeprecated/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py:603` 的 `_step()`；普通路径调用 `self._physics_sim_interface.simulate(self.get_physics_dt(), current_time)`。
- [train_pgmt_stage2.py](../scripts/train_pgmt_stage2.py) 的逐步全量奖励断言确有同步开销，但本次直接计时约 0.03 秒，不能解释 80–100 秒采样。先前提出的“审计可能主要拖慢 collection”属于未验证猜测，本次测量不支持它。
- 800 列接触过滤矩阵存在规模成本，但传感器读取调用没有表现为本次主耗时；不能据此宣称原生模拟内部维护接触过滤完全没有成本。
- 27 个奖励入口共享 `PGMTState.rewards()` 的控制步缓存，没有执行 27 份完整奖励计算。
- 动作存储为 `cuda:0`，地形高度查询为 GPU Warp，headless 无渲染；当前没有证据支持把磁盘读取、CPU 动作搬运或网络优化作为主要瓶颈。
- 系统 governor 名称是 `powersave`，现场 CPU 频率仍有 4–5 GHz；不能仅凭启动警告认定 CPU 固定低频。

后续应优先细分原生 `simulate` 内部时间，核对当前 Isaac Lab/Isaac Sim 6 兼容路径、碰撞过滤及共享地形规模带来的影响，再做受控对照。不要为此直接降低物理频率、改变接触定义或关闭奖励。此次未实施性能修复，也未恢复正式训练。

## 证据

- [原始墙钟计时](../logs/pgmt_collection_profile/wall_8192/wall_times.json)
- [诊断结论与正式 checkpoint 哈希](../logs/pgmt_collection_profile/conclusion.json)
- [独立运行状态](../logs/pgmt_collection_profile/wall_8192/training_status.json)
- [诊断计时器](../logs/pgmt_collection_profile/wall_timers.py)
- [独立运行入口](../logs/pgmt_collection_profile/profile_wall_once.py)
- [保留但不用于占比的 cProfile 输出](../logs/pgmt_collection_profile/native_8192/collection_profile.txt)

## 与原生 PMT 的物理配置对照（2026-10-15）

不是只有 `envIdInBoundsBitCount: 4 → 0`。对照正式 PGMT 的 `params/env.yaml`、原生 PMT perceptive smoke 的实际落盘配置、Stage1 8192 环境配置，以及当前源码，结果如下。不同规模的历史运行只用于配置核对，不作为吞吐量对照。

| 项目 | 原生 PMT | 当前 PGMT Stage2 |
| --- | --- | --- |
| `physxScene:envIdInBoundsBitCount` | 原生 Cloner 设置 4 | Cloner 完成后覆写为 0；环境碰撞过滤仍开启 |
| `sim.enable_scene_query_support` | headless 默认 false，原生与 Stage1 落盘配置均 false | 训练入口无条件设为 true |
| `sim.physx.gpu_collision_stack_size` | `2**26` = 64 MiB | `2**28` = 256 MiB |
| 地形碰撞体 | 原生平地为 plane；原生 perceptive 导入单一合并 mesh | v6 USD 保留 800 个 patch mesh + 4 个边界 mesh |
| 接触传感器 | 全身 `contact_forces` | 保留全身 sensor，增加双脚各一个 terrain-filtered sensor，每个列出 800 个 collider |
| 机器人加载 | 原生 G1 URDF 转换导入 | 显式加载已准备的 `native_robot/main.usd`；配置中的 rigid/articulation/actuator 属性与原生相同 |
| 启动环境 | 本机 `pmt` 环境安装 Isaac Sim 5.1.0.0，通用入口不含 Sim 6 bridge | 专用 shell 固定 `isaacsim` 环境，安装 6.0.1.0，并启用兼容桥 |

仅比较落盘 `sim` 字典，差异恰好是 scene query 开关和 collision stack 两项；`envIdInBoundsBitCount` 是 USD 运行期覆写，不在该字典内。实际比对文件：[native_pmt_config_comparison.json](../logs/pgmt_collection_profile/native_pmt_config_comparison.json)。

以下保持一致：`sim.dt=0.005`、`decimation=4`、控制步 0.02 s、TGS solver、机器人 solver position/velocity iterations=8/4、self-collision 开启、`use_fabric=True`、`replicate_physics=True`、`filter_collisions=True`、`lazy_sensor_update=True`、`clone_in_fabric=False`、`env_spacing=0`。机器人刚体/关节根属性与 actuator 配置比对相同。两者使用原生 `ManagerBasedRLEnv.step()` 的四子步循环，PGMT 没有替换积分器。

另外，PGMT 的地形 reset 使用刚体放置后 `sim.forward()` 刷新运动学；原生 PMT 地形任务关闭的材质/COM/推力随机化，PGMT 继承平地任务保留。这些影响状态或工作量，不能称为完全相同的物理场景。上一轮 24 步诊断没有 reset，故 reset 刷新不是该轮 79 秒的来源。

最值得优先做单变量验证的是 `enable_scene_query_support`：本地 Isaac Lab 明确说明此开关创建 scene query manager，会增加开销，GUI 会自动打开；当前 headless 训练入口却主动开启。PGMT 高度扫描和落脚查询使用 Warp mesh，不是 PhysX scene-query 接口。**这提供了明确的可疑差异，但本轮尚未切换该开关实测，不能宣称已经证明它是瓶颈。** 不应仅据当前计时把慢速归咎于 `bits=0`，也不应直接恢复 bits=4 导致已验证的大额碰撞缓存分配问题。

本次仅检查并记录差异，未修改训练配置或重新启动仿真。

## 关闭 scene query 的单变量测试（2026-10-15）

按用户要求，独立入口仅把 `enable_scene_query_support` 改为 `False`，执行 CUDA 0、8192 环境、headless 的 3 次 PPO 更新，随后正常退出。正式训练仍停止在 383 次更新；生产 `train_pgmt_stage2.py` 未修改。本节是前述“尚未做开关 A/B”之后的新证据。

该字段参与训练契约严格校验，因此没有跳过 checkpoint 校验：从正式 `final.pt` 制作 `query_off_input.pt`，只改 `training_contract.runtime.environment.sim.enable_scene_query_support` 一处元数据。独立递归比较确认包括模型、Adam、课程、动作采样、RNG 和计数在内的其余字段完全一致。测试随后对这个明确记录的配置变体执行正常严格恢复。正式 checkpoint SHA256 始终为 `a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`。

落盘配置与前一轮 query-on 诊断逐字段对比，唯一差异为上述布尔值。运行前后通过 `get_enable_scene_query_support()` 读回 `False`；其余 dt、decimation、bits=0、机器人、碰撞体、传感器和网络配置不变。

| 配置/轮次 | Collection | Returns + PPO |
| --- | ---: | ---: |
| 开启，之前测量的首轮 | 80.63 s | 2.50 s |
| 关闭，第 1 轮 | 87.95 s | 4.08 s |
| 关闭，第 2 轮 | 104.04 s | 4.17 s |
| 关闭，第 3 轮 | 112.28 s | 4.12 s |

关闭后 3 轮均值为 101.42 s，后两轮均值 108.16 s。288 次物理子步合计 247.40 s，平均 0.8590 s/子步、82.47 s/iteration，占 collection 的 81.31%。原生 `simulate` 路径仍然很慢，**关闭 scene query 没有消除主要瓶颈**。基线只测了首个 rollout，关闭组后两轮包含 reset；此外测试启动时还有其他用户 GPU 工作负载，PPO 时间也不同，因此不能把差值直接解释为“关闭导致变慢”或精确估计该开关的收益。

新增定位：关闭组 72 个控制步内有 40 次批量 reset（154 个环境 episode 结束）。Reset 合计 41.02 s，其中 `sim.forward()` 40 次、合计 30.15 s，平均 0.754 s/次；这解释了部分后续轮次耗时增加。此前开启组的首个 24 步没有 reset，因此不能据它排除长期训练中的 reset 开销。

功能检查通过：

- 3 次更新/589,824 transitions；27 项奖励、总奖励及所有 loss 有限，四头奖励和总奖励最大误差 `1.91e-6`。
- 检查时接触/承重样本有效率均为 100%；7514 只脚有接触、7423 只脚承重，双脚地形过滤仍各 800 列。
- 扫描形状 `[8192,441,3]` 正常；从 4096 个均匀分布的采样点中取出 4085 个有效地图点，与 Warp 地形网格高度查询的最大差值为 `4.29e-6 m`。
- 检查在计时结束后执行，不增加上述 rollout 计时。测试输出 `final.pt` 的 next_iteration=386，仅属于诊断分支；正式模型仍为 383。
- Isaac 测试进程已退出并释放 GPU，未恢复正式长训。

材料：[汇总报告](../logs/pgmt_collection_profile/query_off_report.json)、[真实开关/接触/地图检查](../logs/pgmt_collection_profile/query_off_8192/scene_query_checks.json)、[墙钟计时](../logs/pgmt_collection_profile/query_off_8192/wall_times.json)、[配置唯一差异](../logs/pgmt_collection_profile/query_off_config_diff.json)、[测试输入元数据变更记录](../logs/pgmt_collection_profile/query_off_input_change.json)、[测试入口](../logs/pgmt_collection_profile/profile_query_toggle.py)。

## “平地对照”的范围澄清

已有 `flat_160` 不是只保留 plane 的原生 PMT 物理场景。`train_pgmt_stage2.py` 在构建完整 v6 场景后，才将 `terrain_types` / `terrain_levels` 设为 0 并更新出生位置；800 个小块与边界碰撞体仍然存在，过滤传感器也保留。其 collection 8.74 s 与 `short_160` 的 10.56 s 不能作为“已排除复杂地形场景开销”的证据，也不能直接与 8192 环境比较。

现有证据只定位到原生物理推进和 reset 刷新，不足以确认主要原因是地形碰撞体组织、额外接触视图、环境过滤，还是 Isaac Sim 6 的运行/兼容路径。下一步应在同一版本、机器人数量和时间步下做真正 plane 场景与 v6 场景的物理推进对照，并区分接触传感器配置的影响；必要时再对同一场景比较原生 PMT 运行栈。此次没有启动新的测试。

## `envIdInBoundsBitCount=4/0` 性能对照（2026-10-15）

按用户要求检查是否为 `4 → 0` 导致 collection 变慢。独立诊断入口复用未修改的正式 Stage2 训练程序，CUDA 0、headless、160 环境、完整 v6 地形、四头策略、seed=722，均从同一个 Stage1 `model_15000.pt` 初始化；scene query 保持正式入口的 `True`。按 **4 → 0 → 0 → 4** 顺序运行四个独立进程，每次 3 轮 PPO，共 12 轮；每轮 24 个控制步、96 个物理子步。采用 160 环境是因为 bits=4 在大规模完整地图上已验证会触发大额显存分配。

四组落盘环境配置完全相同，唯一实验变量通过原有 `--env_id_bits` 接口传入；agent 配置也仅训练契约中的该位数不同。首次 rollout 前保存并逐项比较 root 状态、关节位置/速度、动作 ID/帧号、地形类型/难度，**四组全部精确一致**；独立读取 `initial.pt` 确认初始模型参数逐项精确一致。属性读回均为 `broadphaseType=GPU`、`enableGPUDynamics=True`，环境碰撞过滤保留。每个进程确实完成 288 次物理推进，无 render，所有原有奖励/损失/梯度断言通过。

| 设置 | 物理子步均值，两个进程各 288 次 | Collection 均值，两个进程各 3 轮 | 初始化候选碰撞对 | 初始化 narrowphase heap |
| --- | ---: | ---: | ---: | ---: |
| bits=4 | 62.192 ms | 11.052 s | 3,730,880 | 6.176 GiB |
| bits=0 | 36.546 ms | 9.135 s | 4,160 | 1.053 GiB |

本次 bits=0 物理推进耗时降低 **41.2%**，collection 降低 **17.3%**；两个重复进程的物理子步范围分别为 bits=4 的 60.03–64.36 ms、bits=0 的 35.84–37.26 ms。首次 rollout 均无 reset，物理子步均值仍分别为 61.02 / 35.48 ms。后续每个进程有 8 次批量 reset；bits=4 的 episode 结束数为 10，bits=0 为 8，因此后续状态不能假定逐步完全一致。第一轮结束时候选对分别为 3,731,887 / 3,092，候选对减少在真实 rollout 中仍然成立。

**结论：这组受控测试不支持“bits=0 关闭并行、造成采样变慢”，在当前完整地形的小规模对照中它反而更快。** 不应直接恢复 bits=4。这个参数控制 GPU broadphase 包围盒的环境 ID 位编码，环境 ID 过滤与 GPU dynamics 是独立机制；共享地形在 bits=4 下被扩张为覆盖全空间的包围盒，产生大量远处 mesh 候选对。官方说明见 [IPhysxReplicator](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physx/docs/api/python.html#omni.physx.bindings._physx.IPhysxReplicator.replicate)，本地固定版本源代码见 `logs/pgmt_physx_diagnosis/physx_110_1_source/physx_broadphase_110_1.cu:96`（包围盒编码）和 `:1642`（独立环境过滤）。

限制：160 环境不是 8192 环境的直接 A/B；机器同时存在其他用户的 GPU 负载，反向重复可以减轻顺序偏差，但不等于独占硬件的精确性能基准。绝对秒数及百分比只描述本次测试，不能直接外推 8192 的加速幅度。正式训练保持停止，正式 `final.pt` SHA256 再次核对为 `a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`。

材料：[汇总与状态一致性检查](../logs/pgmt_collection_profile/env_bits_abba/report.json)、[启动脚本](../logs/pgmt_collection_profile/run_bits_abba.sh)、[只读统计探针](../logs/pgmt_collection_profile/env_bits_checks.py)。每个子目录保留 `bits_evidence.json`、`wall_times.json`、`initial_physical_state.pt`、`progress.jsonl` 和独立训练状态。

### 8192 环境原生分项：主要时间位于状态同步

为避免仅凭 160 环境推断大规模行为，另外保持正式配置（bits=0、scene query=True），从正式 383 更新 checkpoint 严格恢复一个独立 8192 环境进程，执行 1 次更新。使用本机 `IPhysxBenchmarks.enable_profile(True)` 和 `subscribe_profile_stats_events()` 记录原生分项，配合既有墙钟计时。生产源码和 Fabric 设置均未修改。

共收到 96 个回调。第一个回调同时携带初始化/复制阶段积累的统计以及 3 组 Update 记录，不能计入稳态均值；保留原始数据，整帧剔除后使用其余 95 个回调，各包含一组 Update 与 Transforms。

| 原生分项 | 样本数 | 平均每物理子步 | 中位数 |
| --- | ---: | ---: | ---: |
| `PhysX Update` | 95 | 104.343 ms | 107.368 ms |
| `Physics Update Transforms` | 95 | **763.776 ms** | 761.348 ms |

Transforms 占两项合计 **87.98%**。外围 96 次 `sim.step` 合计 83.476 s，collection=88.879 s，returns/PPO=4.180 s；无 reset、无 render，奖励/损失审计通过。这是带原生 profiler 的诊断结果，包含 profiler 开销，不与先前无 profiler 的 80.63 s 直接作为加速 A/B 比较。它把先前混在 `simulate()` 内的主要耗时进一步定位到了状态同步阶段；没有把 `PhysX Update` 进一步拆成 broadphase/narrowphase/solver。

运行时读回：

```text
GPU dynamics / GPU broadphase                 True / GPU
physxScene:envIdInBoundsBitCount                0
/physics/fabricEnabled                         True
/physics/fabricUseGPUInterop                   False
/physics/fabricUpdateTransformations           False
/physics/fabricUpdateVelocities                True
/physics/fabricUpdateJointStates               True
/physics/updateToUsd                           False
/physics/updateVelocitiesToUsd                 False
/physics/suppressReadback                      True
```

这与 [Isaac Sim 官方仓库 issue #740](https://github.com/isaac-sim/IsaacSim/issues/740) 的同版本问题吻合：反馈中的 CPU Fabric 同步路径以 `updateTransforms || updateVelocities` 为条件，**只关闭 transform 更新仍会执行完整更新路径**；启用 GPU interop 或同时关闭两项同步被列为处理方向。本机已确认对应运行设置与主要耗时分项，但本轮没有测试这些开关的修复效果，也没有禁用同步。下一步优先受控测试 `fabricUseGPUInterop=True`，保留正确同步，再核验位姿/速度、地形扫描、接触奖励和 reset；如考虑关闭同步，需要另行验证其语义与可视化影响，不能直接用于正式训练。

综合结论：**`bits=0` 不是现有证据支持的主要瓶颈；当前最大的耗时位于 Fabric 状态同步。** 160 环境直接 A/B 中 bits=0 还降低了物理耗时。未在 8192 环境强行恢复 bits=4，因为此前已验证其共享地形候选对膨胀与 OOM；不能由此声称 bits=0 在任何规模上都没有 broadphase 成本。

本次 4 个小规模进程和 1 个 8192 分项进程均完成并退出。8192 诊断输出的 `next_iteration=384` 仅属于独立诊断分支，不替代正式 383 更新模型。材料：[原生分项与运行设置](../logs/pgmt_collection_profile/physx_zones_8192/physx_zones.json)、[外围计时](../logs/pgmt_collection_profile/physx_zones_8192/wall_times.json)、[启动脚本](../logs/pgmt_collection_profile/run_physx_zones.sh)、[初始模型/agent 配置核对](../logs/pgmt_collection_profile/env_bits_abba/initial_model_check.json)。

## Fabric GPU interop 验证（2026-10-15）

用户要求执行上述验证。独立进程复用正式 Stage2 入口，从正式 383 更新 checkpoint 严格恢复模型、Adam、课程和动作 sampler；保持 CUDA 0、8192 环境、bits=0、scene query=True、headless、全部奖励和 PPO 参数。只在场景建立后、首次物理 reset 前设置 `/physics/fabricUseGPUInterop`，不修改生产训练源码。此全局设置不在旧 checkpoint 的训练契约内，本次通过独立目录和显式运行设置记录该实验差异；正式接入时应补充运行元数据/续训契约，不能只依赖旧配置字典记录它。

### 显存问题与有限重测

1. 默认分配器的 `fabric_interop_on_8192` 完成首轮更新，collection=14.844 s，原生 Transforms 平均约 9.98 ms/子步；第二轮 PPO `loss.backward()` 申请额外 3 GiB 时 OOM。报错时 PyTorch 已分配 11.80 GiB、预留未使用 3.56 GiB、整卡空闲 2.35 GiB。因此首轮可作为性能线索，但整个运行未通过验收。
2. `PYTORCH_ALLOC_CONF=expandable_segments:True` 重测在环境初始化、动作库加载后原生崩溃，guard 退出码 245（worker SIGSEGV）。Isaac AppLauncher 的信号退出又触发递归异常，原生根因未确认；不能把这次崩溃认定为 PPO 问题，也不能声称已证明 allocator 的具体兼容性原因。该运行没有完成更新，不用于性能比较。
3. 改为原生分配器的 `PYTORCH_ALLOC_CONF=max_split_size_mb:512`，避免拆分大于 512 MiB 的块；开启组完成 3 次更新和全部后置功能检查，环境数仍为 8192。这个参数用于缓解大块显存碎片，语义见 [PyTorch 分配器说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)。这里只证明本次有限重测通过，不宣称长期 OOM 风险已经消失。

关闭 interop 的匹配基线也使用相同 `max_split_size_mb:512`，避免将 allocator 的影响误算为 interop 收益。每组运行 3 次更新：第 1 次使用原生 profiler；后 2 次关闭 profiler，仅保留轻量墙钟计时。原生分项剔除首回调中的初始化累计数据。状态/扫描/reset 检查在训练和诊断 `final.pt` 保存之后执行，不进入 collection/PPO 计时，也不改写该 checkpoint。

两组落盘 env/agent 配置完全一致；初始模型以及 root 状态、关节位置/速度、动作 ID/帧、地形类型/等级精确一致。已完成的开启组功能检查包括：

- 27 项奖励、四头合计、PPO loss/梯度检查通过；独立重读诊断 `final.pt`，模型和 141 个优化器状态张量有限，next_iteration=386。
- 对照 PhysX 原始张量检查全部 8192×30 个刚体的位姿和 COM 速度，训练后、每次指定 reset 后及 8 个额外控制步后共检查 12 次，最大误差均为 0。
- 五种地形在 0/5/9 难度上各选 8 个环境，累计 120 次 reset。全身参考位置最大误差 `2.29e-5 m`；根部线/角速度及关节速度误差均为 0，起始帧和离地间隙检查通过。
- 上述位置按实际 torso/yaw 独立重建 21×21 查询网格，扫描高度最大误差 `6.68e-6 m`，地图有效掩膜精确一致；进入网络前无效观测仍为 -3。
- 训练后接触/承重有效率 100%，双脚各 800 个地形过滤列保留；额外控制步中参考变换对存活回合保持不变，参考接触帧号与 command 一致。

材料：[测试入口](../logs/pgmt_collection_profile/profile_fabric_interop.py)、[运行时设置与功能探针](../logs/pgmt_collection_profile/fabric_interop_checks.py)、[可复现启动脚本](../logs/pgmt_collection_profile/run_fabric_interop.sh)、[失败与成功尝试记录](../logs/pgmt_collection_profile/fabric_interop_attempts.json)、[初始状态/模型一致性](../logs/pgmt_collection_profile/fabric_interop_initial_comparison.json)。

### 匹配对照最终结果

开启组和关闭组均完成 3 次更新、后置功能检查，并正常退出；未降低 8192 环境数。两组运行时设置逐字段对比，唯一差异为 `/physics/fabricUseGPUInterop: False → True`。以下采用各组后两轮的**无原生 profiler**均值：

| 指标 | interop=False | interop=True | 结果 |
| --- | ---: | ---: | --- |
| Collection / iteration | 110.135 s | **14.716 s** | **7.48×** 吞吐提升，耗时降低 86.6% |
| 96 次物理推进 / iteration | 83.379 s | **4.104 s** | **20.32×** |
| 单个 physics substep | 868.533 ms | **42.750 ms** | 同上 |
| reset 中 `sim.forward()` 平均每次 | 767.747 ms，42 次 | **10.239 ms**，39 次 | 每次约 75×；reset 次数不完全相同 |
| Returns + PPO / iteration | 4.033 s | 4.066 s | 基本不变 |

Collection + returns/PPO 从约 114.17 s 降至 18.78 s，约 **6.08×**；不包含轮末日志/保存。第一轮单独使用原生 profiler 的分项也吻合：`Physics Update Transforms` 从 **760.012 ms 降至 10.270 ms**；`PhysX Update` 为 105.804 / 91.966 ms。带 profiler 和无 profiler 的物理时间不可混用，性能结论使用上表。

两组后置功能检查均通过，模型和 141 个优化器张量独立重读有限；两组诊断 checkpoint 都为 next_iteration=386。开启组 Torch peak allocated/reserved 为 15158.57 / 16288 MiB，关闭组为 15156.58 / 16256 MiB，未发生新的 OOM。这些是 Torch 峰值，不是整卡总显存。

验收边界：

- 这是每组 3 次更新的 headless 短训验证，不是长期稳定性证明；同卡仍有其他用户的计算进程，绝对秒数包含共享硬件波动。
- 初始状态/模型精确一致，但末态不保证逐位相同。本次 72 控制步内 done 数为开启组 147、关闭组 137，最终模型张量最大绝对差为 0.01253；没有通过重复同开关运行去分离 GPU 非确定性与设置的潜在数值影响。因此不把本次功能与数值验收描述为“逐步轨迹完全相同”。
- 验证的是 PhysX 张量、训练观测/奖励、扫描和 reset 的正确性。两组都保持原来的 `fabricUpdateTransformations=False`，本次没有打开 GUI 验证渲染，也没有宣称 Fabric 中关闭更新的位姿会刷新。
- 本次通过的组合是 **interop=True + 原生 allocator `max_split_size_mb:512`**。生产入口仍未修改，正式训练保持停止，正式 checkpoint 仍为 383 更新，SHA256=`a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`。

结论：GPU interop 在本机 8192 环境下显著消除了此前的 Fabric CPU 同步瓶颈，并通过本次有限功能验收。后续正式接入应同时记录显存分配设置与运行时开关，并继续监测长训的显存和速度。

最终材料：[完整对照汇总](../logs/pgmt_collection_profile/fabric_interop_report.json)、[开启组功能结果](../logs/pgmt_collection_profile/fabric_interop_on_split512_8192/interop_functional_checks.json)、[关闭组功能结果](../logs/pgmt_collection_profile/fabric_interop_off_split512_8192/interop_functional_checks.json)。复现命令（第二个参数必须使用新的目录名）：

```bash
PYTORCH_ALLOC_CONF=max_split_size_mb:512 \
  bash logs/pgmt_collection_profile/run_fabric_interop.sh 1 fabric_interop_on_recheck
PYTORCH_ALLOC_CONF=max_split_size_mb:512 \
  bash logs/pgmt_collection_profile/run_fabric_interop.sh 0 fabric_interop_off_recheck
```

## 2026-10-15：pmt / Isaac Sim 5.1.0 原生训练测试

按用户要求，使用 `/home/lzp/miniconda3/envs/pmt/bin/python`，原生 `isaaclab.python.headless.kit`，不启用 Sim 6 兼容桥。CUDA 0、8192 环境、完整 v6 地形、bits=0、scene query=True、Fabric=True / GPU interop=False、seed=722，从正式 383 更新 checkpoint 的兼容副本严格恢复，完成 **3 次 PPO 更新**，独立测试 checkpoint 为 next_iteration=386。测试及功能检查正常退出（exit 0），未 OOM、未减少环境数量；正式训练仍停在 383 更新，未覆盖其 checkpoint，也未改变生产启动脚本。

### 环境和 checkpoint 兼容处理

- 实际版本：Python 3.11.16、Isaac Sim 5.1.0.0、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128、NumPy 1.26.0、Warp 1.12.1。
- 直接读取 Sim 6 环境保存的文件会报 `ModuleNotFoundError: numpy._core`。在独立转换进程中，用 scoped unpickler 将 `numpy._core` 映射到 NumPy 1 的 `numpy.core`，再另存为 `logs/pgmt_collection_profile/formal_383_numpy1.pt`；没有改安装包或原 checkpoint。
- 在原保存环境独立加载两份文件，核对 **211 个 tensor、1 个 ndarray、6814 个标量叶节点全部相等**，包括 optimizer、课程、sampler 和 RNG；训练入口的严格续训校验也全部通过。
- PyTorch 2.7 使用 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`。与上一轮 Sim 6 测试使用相同的 native allocator 分割限制；变量名称采用该版本支持的名称。
- 与 Sim 6 两组对比，序列化 env/agent 配置、初始模型、root/joint 状态、动作 ID/帧、地形等级/类型均完全相同。

### 性能结果

首轮启用原生 PhysX profiler，collection 为 8.507 s；后两轮关闭该 profiler，collection 分别为 **11.271 / 14.138 s**。下表统一使用后两轮平均值，仍包含相同的轻量 wall timer 和训练审计。

| 指标 | Sim 6 / interop=False | Sim 6 / interop=True | pmt Sim 5.1 / interop=False |
|---|---:|---:|---:|
| Collection / 轮 | 110.135 s | 14.716 s | **12.704 s** |
| 96 次 `sim.step` 合计 / 轮 | 83.379 s | 4.104 s | **3.207 s** |
| 平均物理子步 | 868.533 ms | 42.750 ms | **33.407 ms** |
| Reset `sim.forward` 均值 | 767.747 ms | 10.239 ms | **1.986 ms** |
| Returns + PPO / 轮 | 4.033 s | 4.066 s | **3.745 s** |

Sim 5.1 collection 比上一轮 Sim 6 interop=False 测试约快 **8.67 倍**，比 interop=True 测试约快 **1.16 倍**。Torch 峰值 allocated=15155.55 MiB、reserved=16288 MiB；结束时全卡显存快照为 26573 MiB，包含同卡其他进程，并非本进程峰值。该结果来自短训且后两轮仍在变化，不能当作长期稳态吞吐。

实际运行设置还显示：Sim 5.1 原生 headless 的 `fabricUpdateTransformations / fabricUpdateVelocities / fabricUpdateJointStates` 均为 False；先前 Sim 6 为 False / True / True。首轮去掉首次累计回调后的原生 profile：`PhysX Update` 平均 36.351 ms，`Physics Update Transforms` 仅 **0.00110 ms**。这与此前 Fabric CPU 同步瓶颈的判断一致；bits 仍为 0，完整地形和 8192 环境仍能快速运行。**这不是只切换 Isaac Sim 版本的单变量实验**：Python/Torch/NumPy、experience 和 Fabric 默认设置也随环境改变；不能把全部差异单独归因于版本号。

### 功能、数值和日志检查

- 72 个控制步、27 项 reward、四头加总校验通过，最大加总误差 `9.54e-7`；发生 164 次 done，未出现非有限 loss/梯度，selector、patch encoder、IFM、actor、critic 和四个 value head 均有更新。
- 对全部 8192 × 30 个身体实例检查 12 次，机器人状态与直接 PhysX tensor 的位置/姿态、COM 速度最大差均为 0。Fabric 不向渲染侧同步速度/关节状态，没有导致本次训练使用的 PhysX tensor 状态陈旧。
- 五类地形 × L0/L5/L9 × 8 环境，共 120 次受控 reset：参考全身位置最大误差 `2.29e-5 m`，root/joint 速度误差 0，起始帧为 0、放置间隙通过。
- 21×21 terrain map 按实际 torso 位置/yaw 重建，与地形查询最大高度差 `5.72e-6 m`，有效掩膜完全相同，无效网络高度为 −3；原生 scanner 另抽查 4086 个有效点，最大误差 `4.29e-6 m`。
- 地形承重/接触有效率均 100%，左右脚各保留 800 个地形过滤对象；额外 8 步验证参考变换保持、接触标签随 motion/frame 更新、机器人状态确实变化。
- 独立使用 **pmt** CPU 重读测试 `final.pt`：next_iteration=386，模型及 141 个 optimizer tensors 有限，课程/sampler/RNG 已保存。该文件是测试产物，未替换正式 383 更新文件。
- TensorBoard 文件包含 **1078 个 scalar tags**，27 项 `RewardStep` 均有 iteration 383/384/385 数据且数值有限，包含四头 loss、课程、采样器等统计。现有 6006 服务仍只指向原正式训练目录；本次测试日志需另选下方目录查看。

运行目录：`logs/pgmt_collection_profile/pmt_native_8192/`。材料：[汇总报告](../logs/pgmt_collection_profile/pmt_native_report.json)、[完整训练日志](../logs/pgmt_collection_profile/pmt_native_8192.log)、[checkpoint 转换核验](../logs/pgmt_collection_profile/native_pmt_checkpoint_conversion.json)、[功能检查](../logs/pgmt_collection_profile/pmt_native_8192/interop_functional_checks.json)。复现同样的有限测试（使用新目录名）：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash logs/pgmt_collection_profile/run_pmt_native.sh pmt_native_8192_recheck
```

单独查看本次测试 TensorBoard：

```bash
/home/lzp/miniconda3/envs/pmt/bin/tensorboard \
  --logdir /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_collection_profile/pmt_native_8192 \
  --host 127.0.0.1 --port 6007 --load_fast=false
```

本次通过的是 **headless 有限短训与训练状态正确性**，不代表 GUI 渲染已验证或长期稳定性已验证。测试 worker 已退出并释放 GPU；正式 checkpoint SHA256 仍为 `a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`。

## Sim 5.1 剩余 collection 开销与优化顺序（实施前的代码审查）

以下保留当时的候选分析；2026-10-15 的逐项实施、实测收益、正确性和续训检查见 [collection 优化实施与验收](PGMT_COLLECTION_OPTIMIZATION.zh-CN.md)。

重新分析上述原生测试 iteration 384/385 的 wall timers，两轮 collection 平均 12.704 s。下表选取相互不嵌套的主要调用段；未列出的调用及 runner 开销为余量。计时为主机 wall time，包含调用位置等待先前异步 CUDA 工作的时间，不能当作各 GPU kernel 的独立耗时。

| 部分 | 平均每轮 | Collection 占比 |
|---|---:|---:|
| 环境 reset 全链路 | 4.669 s | 36.7% |
| 96 次物理推进 | 3.207 s | 25.2% |
| RewardManager（含接触计算） | 1.486 s | 11.7% |
| Scene 写入仿真（含 actuator） | 1.024 s | 8.1% |
| Scene 更新（含每子步全身 contact sensor） | 0.781 s | 6.1% |
| step 包装器自身（含奖励审计） | 0.329 s | 2.6% |
| 策略 act | 0.235 s | 1.8% |
| 观测计算（含 height scanner） | 0.220 s | 1.7% |

第二/三轮分别出现 15/24 批 reset，reset 合计从 3.364 s 增至 5.973 s，collection 从 11.271 s 增至 14.138 s；多出的 reset 开销解释了其中约 2.61 s 的增长。reset 内 `_reset_robot_state` 平均 1.584 s，reset 排除已包裹子调用后的剩余部分约 3.013 s；目前尚未将这 3.013 s 精确分配给 scene reset、各 manager、采样和日志同步，不能直接断言全部是某一个 `.item()` 或放置检测。`sim.forward` 仅约 2 ms/次，已经不是 reset 的主要问题。

建议按以下顺序验证，所有条目目前均为候选优化：

1. **细分 reset 后优化统计同步和采样**。原生 command/curriculum/termination reset 含逐项 `.item()`；`AdaptiveSampler.get_metrics()` 还将统计读回 CPU，再由 `_update_metrics()` 扩展为 8192 元素 tensor。将用于日志的量在 GPU 上收集、批量读回，保留旧 episode 的统计快照和准确结算顺序。`sample_motion_starts()` 的 `unique → 分组 nonzero → multinomial` 也值得改为更少同步的批量采样；须验证地形兼容和动作级概率分布，不能缓存已经变化的 adaptive 权重。
2. **减少放置与 reset 的重复工作**。当前固定第 0 帧，可以缓存每个动作的源姿态、源碰撞球中心、未来帧位置；随机 XY/yaw 和实际地形校验仍每次计算。`place_on_surface()` 目前执行完整参考旋转，未来越界检查也旋转姿态/速度但只使用位置，可改为位置专用运算。基类 reset 还会对所有 8192 环境调用 `_compute_relative_poses()` / `_update_metrics()`，可评估只更新 reset 子集。保留不穿透/越界检查、参考统一变换及旧奖励与 sampler 的结算语义。
3. **清理 PGMT 未使用的原生接触历史**。实际全身 `contact_forces` 配置仍为 `history_length=3, track_air_time=True`，因此每轮更新 96 次；当前 PGMT 奖励读的是 `net_forces_w`，其接触事件/时间由自己的 tracker 维护。核验全调用链后可关闭未使用的原生 history/air-time，让当前力按控制步访问时更新。该 sensor 当前平均约 0.435 s/轮，收益空间小于 reset；不可删除地形过滤承重 sensor，也不能漏掉 reward 真正需要的历史。
4. **生产训练降低审计同步频率，再评估 reward 算子融合**。`train_pgmt_stage2.py` 每控制步执行多次 `assert_close`、finite 检查和标量计数；可提供独立验收模式及按 PPO 周期批量校验的生产模式，27 项 reward 和课程日志照常保存。包装器自身当前约 0.329 s/轮，不能靠这一项解释全部 collection。reward 数学核可进一步评估算子融合/编译；`PGMTState.rewards()` 已按控制步缓存，计数 648 次是 27 项 × 24 步的缓存访问，不是整套奖励重算了 648 遍。

CUDA `.item()`、`nonzero()` 和依赖 CUDA tensor 的 Python 条件可能引入主机同步，参考 [PyTorch 官方性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)。当前 height scanner 约 0.111 s/轮，策略 act 约 0.235 s/轮，优先级低于 reset。仅作量级估算：若 reset 用时减半，其余不变，可省约 2.33 s，使 collection 约为 10.37 s；这是估算，不是已验证收益。下一次性能验收应覆盖较多含 reset 的 rollout，比较 transitions/s，并保持 8192 环境、24 步 rollout、物理步长、奖励/终止和采样语义一致。
