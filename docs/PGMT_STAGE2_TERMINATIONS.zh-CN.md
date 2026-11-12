# PGMT Stage2 终止条件

日期：2026-10-14。按用户要求修改两个 Stage2 任务：`pgmt_stage2` 和 `pgmt_stage2_terrain`。Stage1 与历史 P1 `PGMTPlacementEnvCfg` 的判据保持原状。下面阈值是本次用户指定设置，不标为论文公开参数。

## 1. 当前规则

前四项跟踪失败均在控制步末采样，误差阈值使用严格 `>`，**每项必须连续超限满 0.2 s 才终止**。控制周期为 0.02 s，即连续 10 个控制步；前 9 步不因该项终止，第 10 步触发。root 明确为 `pelvis`；躯干为 `torso_link`。

| term | 判定 | 结果 |
| --- | --- | --- |
| `root_xy` | 实际 pelvis 与本回合统一放置后的参考 pelvis 的 XY 欧氏距离 >1.0 m | 连续 0.2 s 后 `terminated=True` |
| `anchor_pos` | 实际 torso 世界 Z 与“实际 torso XY 的地形高度＋当前动作帧原始 torso Z”的差绝对值 >0.4 m | 连续 0.2 s 后 `terminated=True` |
| `ee_body_pos` | 左右手腕任一相对 pelvis 的高度跟踪误差 >0.4 m；不检查脚踝 | 连续 0.2 s 后 `terminated=True` |
| `anchor_ori` | 原生实际/参考 torso 姿态下的投影重力 Z 分量差绝对值 >0.8 | 连续 0.2 s 后 `terminated=True`，姿态公式保持原样 |
| `time_out` | `episode_length_buf >= 500`，或步数 ≥498 且动作已到末帧；10 s 硬上限、2 控制步边界容差 | `truncated=True` |
| `motion_finished` | 当前动作 segment 到末帧 | 继续沿用既有显式 truncation |
| `map_out_of_bounds` | 实际或参考 tracked bodies 超出有效地图 | 继续沿用既有立即 truncation |

计时按环境、按终止项独立保存；某项在一个控制步恢复正常就将该项连续计数清零，不能累计零散失败，也不能跨项拼成 0.2 s。手腕项仍以“任一手腕超限”的合并布尔条件计时。部分 reset 只清零对应环境；reset 当步重复查询不计入新回合时间，同一 `common_step_counter` 的重复读取不重复计时。计时基于控制步整数计数，避免浮点累加把第 10 步误推迟到第 11 步。不是额外的出生宽限期，也不随地形/难度变化；奖励计算不增加延迟。动作耗尽、地图越界和 10 s 超时仍可在这段容忍期内立即截断。

2026-10-15：按用户选择保留 10 s 上限，新增时间边界容差。**未结束的动作不能因这 0.04 s 容差被提前截断**；规则、接口及验收见第 5 节。Stage1 与历史 P1 仍使用原生 `time_out`。

### 水平轨迹

$$
e_{xy}=\left\|p^{robot}_{pelvis,xy}-\widehat p^{motion}_{pelvis,xy}\right\|_2>1.0\;\mathrm{m}.
$$

这里的参考是 reset 时建立的统一平移/yaw 变换作用于**当前动作帧**的结果。不是 env origin、小块中心，也不是逐步重新对齐到机器人的位置。因此真实轨迹偏离会增长，跨 tile 不重建变换。

### 支撑面相对躯干高度

$$
z^{target}_{torso}=h_{mesh}(x^{robot}_{torso},y^{robot}_{torso})+z^{source}_{torso},\qquad
e_z=\left|z^{robot}_{torso}-z^{target}_{torso}\right|>0.4\;\mathrm{m}.
$$

`z_source` 是当前 NPZ 动作帧在原始平地坐标系中的 torso 高度，排除本回合地形 origin、统一放置 Z 和出生防穿透抬升；不能把已经放置的世界参考 Z 再加一次地形高度。支撑面在此按 torso 当前世界 XY 的竖直射线高度定义，不使用脚接触平均高度或姿态相关的局部地面拟合。

地形高度来自 `terrain.query()` 的干净碰撞网格，不使用有噪声的 21×21 网络地图；保留倒金字塔负高度。跨 tile 后读取当前位置高度，出生课程归属不变。地图外高度不填 0，不产生这一项伪高度失败，交给地图越界项；有效地图内射线缺测报错。非有限机器人位置/原始参考高度不会被静默视为正常。

### 手腕相对根部高度

对左右 `wrist_yaw_link` 各自计算：

$$
e_j=\left|(z^{robot}_j-z^{robot}_{pelvis})-(z^{source}_j-z^{source}_{pelvis})\right|,
\qquad \max_{j\in\{left\ wrist,right\ wrist\}} e_j>0.4\;\mathrm{m}.
$$

两边分别减去各自 pelvis 高度，全身共同升降不改变该误差。源参考的统一 yaw 和 Z 平移会在此消除。脚踝不再参与这项终止，脚部接触奖励仍照常计算。

姿态项的原始超限判定继续使用原函数，外层增加 0.2 s 连续计时；0.8 是重力方向分量差的无量纲阈值，不是 0.8 rad，也不是完整四元数角度误差。

## 2. 生命周期、课程与 checkpoint

三个原始判定函数只读取当前物理状态及 command 当前帧，不推进动作、不改参考变换、不维护另一套 contact 或 outcome 缓存。外层 `SustainedTrackingFailure` 使用原生 `ManagerTermBase` 生命周期管理独立计时；`TerminationManager.reset(env_ids)` 在旧回合课程和 sampler 结算后清零对应计数。重复读取不额外累计，局部 reset 后直接读取新 motion/frame。

跟踪失败仍由原生 TerminationManager 进入 reward → 课程结算 → sampler outcome → reset 链路。失败降一级，提前 timeout 保持等级，完整动作满足原成功条件才升级。若完整动作结束与 10 s 同步，仍按完整动作判据处理；若失败与末帧/越界同时发生，既有课程优先级仍为失败优先。`terminated` 与 `truncated` 可能同时为真。

动作从第 0 帧开始，10 s 内无法播完的长 segment 会提前 timeout，不能仅凭到达时间上限升级地形。本次不修改动作采样概率规则、动作长度或课程成功标准。

正式 runner 的 `training_contract` 保存实际 `terminations`（包含每项 `duration_s=0.2`、原始判定及其参数）和 `episode_length_s`。**修改前的 Stage2 checkpoint 不属于同配置严格续训**；直接 `--resume` 会被拒绝。仍可用指定 Stage1 `model_15000.pt` 或三头 P2 `--initialize-from` 开始新四头训练；此显式迁移新建优化器与 sampler/课程。同配置续训会重新初始化物理回合，连续计数随 reset 清零，不恢复上一回合的残留计时。此前 P3.4 的六次更新和 checkpoint，以及本页下方即时终止版结果作为历史验收保留。

## 3. 实现位置与验收

- [pgmt_terminations.py](../pmt_tasks/pgmt_terminations.py)：三个原始判定函数及独立控制步计时器。
- [mdp/pgmt_terminations.py](../pmt_tasks/mdp/pgmt_terminations.py)：原生 manager 有状态终止项，负责调用原判定及按环境 reset 计时。
- [PGMTTerrainMotionCommand.source_body_pos](../pmt_tasks/mdp/commands/pgmt_motion_command.py)：与当前 motion/frame 同步、未经放置变换的原始 tracked-body 坐标。
- [Stage2 环境配置](../pmt_tasks/env_cfgs/pgmt/stage2.py)：接入 root XY、地形相对 torso、相对 pelvis 手腕及 10 s；四头环境继承此配置。
- [test_pgmt_terminations.py](../tests/test_pgmt_terminations.py)：CPU/CUDA 边界、负高度、当前 XY、平移抵消、脚踝排除、根部命名、切帧/reset/重复读取、地图无效及非有限输入。
- [validate_pgmt_terminations.py](../scripts/validate_pgmt_terminations.py)：真实 Isaac 五类十级、原始 NPZ/NumPy 对照、物理根部偏移及真实手腕/参考帧失配、跨 tile、500 步超时、课程和 sampler 只结算一次。

有限验收入口：

```bash
bash scripts/validate_pgmt_terminations.sh --log_dir logs/pgmt_p3_terminations/isaac_next
```

验收只在有限步探针中手动设置姿态和时间计数，不等同策略自主完成高难度地形或连续存活 10 s。

### 当前 0.2 s 连续容忍时间验收

- **98 项 CPU/CUDA 测试通过**：连续第 10 步触发、恢复正常清零、四项/多个环境独立计时、部分 reset、同控制步重复读取、不跨未采样间隔累计，以及既有阈值、课程、奖励和 checkpoint 回归。见 [cpu_cuda_tests.txt](../logs/pgmt_termination_delay/cpu_cuda_tests.txt)。
- **真实 Isaac：5 环境、53 个控制步、五类十级通过**。两个 Stage2 配置的四项均为 `duration_s=0.2`，Stage1 原判据不变。对每个跟踪项，在每个控制步前保持受控物理姿态制造超限：前 9 步继续，第 10 步真实终止、降级并 reset，四项计数均清零。这是计时/lifecycle 探针，不代表策略自主保持该姿态。
- 地图越界、动作耗尽保持单步 truncation；计数从 499 到 500 的超时也立即截断且课程等级保持。11 次 done 对应 11 次 sampler outcome，254 次存活参考变换检查、265 次接触标签刷新检查通过。见 [training_status.json](../logs/pgmt_termination_delay/isaac_01/training_status.json) 和 [measurements.json](../logs/pgmt_termination_delay/isaac_01/measurements.json)。
- 本次没有启动新的 PPO 训练，验收 Isaac 已关闭。下方旧 PPO/续训结果属于即时终止版，不作为本次持续计时版的训练性能或续训实测结果。

### 历史独立验收结果（未添加 0.2 s 连续计时）

- **82 项 CPU/CUDA 测试通过**，包含新终止数值测试、既有 MDP/课程/配置/四头迁移与 checkpoint 回归，并验证终止规则或回合长度变化会拒绝严格续训。输出见 [cpu_cuda_tests.txt](../logs/pgmt_p3_terminations/cpu_cuda_tests.txt)。
- **真实 Isaac：5 环境、13 个控制步、50 个地形区域通过**，使用真实 walk1 NPZ 帧及独立 NumPy 判定。实际配置为 7 项终止、10 s/500 步，Stage1 配置仍为 15 s 和原末端集合。[isaac_05/training_status.json](../logs/pgmt_p3_terminations/isaac_05/training_status.json) 为 `passed`。
- 检查了出生 stairs L9 / 当前 slope L3，地形高度随实际位置变化，统一参考和出生分配不变；根部水平偏移触发失败降级；整体抬高 0.6 m 触发 torso 失败，手腕相对高度保持正常；真实初始化手腕与另一原始参考帧产生 >0.4 m 误差，正确触发手腕失败。
- 计数从 499 前进到 500 后真实 timeout/reset，课程保持原级。7 次 done 对应 7 次 sampler outcome，82 次初始化检查、58 次存活参考不漂移检查、65 次接触标签刷新检查通过。原始误差、地面高度、frame 与判定记录见 [measurements.json](../logs/pgmt_p3_terminations/isaac_05/measurements.json)。
- 前四次脚本调试记录保留但不计为通过：前两次是手动 reset/写入姿态位于 inference mode 外，统一验收上下文后解决；另两次所选手臂姿态没有实际产生 >0.4 m 失配，最大仅约 0.387 m，随后改用真实动作帧构造足够失配。未为通过测试降低生产阈值。

### 历史正式四头训练与恢复（未添加 0.2 s 连续计时）

使用指定 Stage1 `model_15000.pt` 显式初始化，在完整 862 动作、v6 地形、160 环境和全部 27 项奖励下完成 **1 次 PPO 更新**，记录于 [train_160_01](../logs/pgmt_p3_terminations/train_160_01/training_status.json)。24 控制步中产生 1 次自然失败，四组和与标量 reward 最大误差 `1.19209290e−7`；四个价值头与感知、IFM、actor 均有梯度及参数更新。

随后新 Isaac 进程从该 final checkpoint 恢复，0 次更新后保存退出，[restore_160_01](../logs/pgmt_p3_terminations/restore_160_01/training_status.json) 为 `completed`。独立读文件核验模型/归一化、Adam、LR、iteration、课程、sampler、实际训练契约和累计训练计数精确一致，恢复采样概率检查通过。新模型 `next_iteration=1`，只用于链路验收。实际旧四头 checkpoint 已验证在修改参数前因训练契约不同而拒绝加载。

首次更新的平均 approximate KL 约 `0.332`、clip fraction 约 `0.572`，自适应 LR 降到 `1e−7`；这次只确认公式、训练及持久化链路，不据此宣称训练稳定或地形完成率提升，也未为此擅自调整 PPO/感知参数。

完整结果见 [acceptance_report.json](../logs/pgmt_p3_terminations/acceptance_report.json)。本次所有 Isaac 进程已退出，没有持续运行训练。

## 4. 2026-10-15：success 非零、各地形 timeout 为零的核查（边界容差修改前）

结论：现有记录符合当前回合定义。10 s / 500 控制步是回合上限；完整动作结束会提前截断，课程 success 不要求达到 10 s。本次只核查并补充说明，没有修改终止、课程、采样或 PPO 行为。

- 正式训练 checkpoint 的 `curriculum.identity.motion_lengths` 中，862 个动作全部为 499 帧，固定从第 0 帧开始。数据集 manifest 及原始 NPZ 一致：50 Hz，时间轴 `0..9.96 s`。源 CSV 为 300 帧、30 Hz，最后时间戳为 `299/30 ≈ 9.9667 s`；转换不外推，因此没有 9.98 s 的第 500 帧。详见 [动作数据说明](PGMT_CONTACT_LABELS.zh-CN.md)。
- `pgmt_motion_finished()` 在 `frame_ids >= motion_length - 1` 时触发。原生环境先检查终止、reset，再推进 command；首次从帧 0 开始的回合最多 499 个控制步，自动 reset 后从帧 1 进入下一步的回合最多 498 步，即约 9.98 / 9.96 s，均早于 500 步上限。失败和地图越界还会进一步缩短回合。
- `Curriculum/{family}/L{level}/timeout` 是累计的互斥课程结局：达到普通时间上限且没有被更高优先级结局覆盖。完整动作结局计为 `success`；失败、地图越界另有分类。`OutcomeWindow/...` 记录当前日志窗口的增量，累计课程数会随 checkpoint 恢复。
- `TerminationCount/time_out` 是原始 500 步上限判定的窗口计数；`TerminationCount/motion_finished` 单独记录动作耗尽。不同终止条件可以同时为真，所以原始 motion_finished 数不必等于课程 success 数。
- 原生接口 `extras["time_outs"]` 实际是 `truncated`，包含动作耗尽、地图越界和时间上限。`reward_group_audit.timeout_events` 累加这个广义截断标志，名称容易与课程 timeout 混淆，不能拿两者直接比较。当前仅澄清口径，保留既有日志键及接口。

核查 [正式训练记录](../logs/pgmt_p4_20261014/formal_8192/progress.jsonl)，383 次 PPO 更新累计结果如下：

| 记录 | 次数 |
| --- | ---: |
| 课程 success | 92,461 |
| 课程 failure | 111,472 |
| 课程 map_exit | 5,253 |
| 课程 timeout / incomplete_motion / other_truncation | 均为 0 |
| 原始 motion_finished | 92,612 |
| 原始 time_out | 0 |
| PPO 广义截断 `reward_group_audit.timeout_events` | 97,890 |

课程结局合计 209,186，与 `done_events` 完全一致。TensorBoard 的 `Train/mean_episode_length` 单位是控制步，最后一个点为 365.48 步，约 7.31 s，是完成回合的滑动窗口均值，并非成功回合的单独时长；同名 `/time` 标签只是横轴改用累计训练时间，纵轴仍是步数。

最近 `pgmt_ppo_scalars_20261015/cuda1_batched` 的 8 次更新总共只有 192 控制步，续训会重置物理回合，因此本次窗口没有新的完整动作 success；图中的非零累计 success 来自恢复的第 383 次 checkpoint。应使用 `OutcomeWindow/.../success` 判断新增成功。

本次 CPU 回归 `tests/test_pgmt_curriculum.py`、`tests/test_pgmt_terminations.py` 共 **28 项通过**。既有真实 Isaac [0.2 s 容忍验收](../logs/pgmt_termination_delay/isaac_01/training_status.json) 的 `timeout_500_neutral_course=true`，已验证 499→500 步触发超时、reset、课程等级保持；本次没有重新启动仿真。无需为了让 timeout 非零而延长、冻结或拼接参考动作。

## 5. 2026-10-15：保留 10 s 上限，增加控制周期边界容差

用户选择保留 10 s，仅调整边界容差。两个 Stage2 任务已将 `time_out` 接入 [pgmt_time_out](../pmt_tasks/pgmt_terminations.py)，配置位于 [stage2.py](../pmt_tasks/env_cfgs/pgmt/stage2.py)：

```python
self.episode_length_s = 10.0
self.terminations.time_out = TerminationTermCfg(
    func=pgmt_time_out, time_out=True,
    params={"command_name": "motion", "boundary_tolerance_steps": 2},
)
```

设控制周期为 Δt、回合控制步数为 n、末帧标志为 M：

$$
N_{\max}=\left\lceil\frac{10}{\Delta t}\right\rceil,\qquad
K=2,\qquad \delta t=K\Delta t,
$$
$$
\mathrm{time\_out}=(n\ge N_{\max})\;\lor\;
\left[M\land n\ge\max(1,N_{\max}-K)\right].
$$

当前 Δt=0.02 s，Nmax=500，容差为 0.04 s。只使用整数步比较，不累加浮点时间。容差单位是**控制步**，不是 0.005 s 的物理子步；接口设为 0 可关闭容差，负数、非整数和布尔值会被拒绝。

| 当前情况 | `time_out` |
| --- | --- |
| n≤497，动作已经耗尽 | False；仍由 `motion_finished` 正常截断 |
| n=498/499，动作已经耗尽 | True；同时 `motion_finished=True` |
| n=498/499，尚未到末帧 | False；允许继续跟踪 |
| n≥500 | True，无论参考是否耗尽 |

两步容差覆盖本批 499 帧动作的 9.96 s 时间轴，以及 Isaac 首次 reset 与自动 reset 后参考推进一拍的差异。直接把全体回合上限改成 498 步，会让首次回合在 frame=497 时丢掉末帧；新增末帧门控避免这个问题。

这项修改仅扩大独立 `time_out` 标记，不增加实际 reset：新增分支已经属于 `motion_finished`，所以总 `done` 和 PPO `extras["time_outs"]` 的并集保持一致。课程继续失败优先、完整动作 success 优先；接近上限的完整动作依然升级一次，不重复累加 timeout 或 sampler。**因此原始 `TerminationCount/time_out` 可非零，而互斥的 `Curriculum/.../timeout` 仍可能为零**，这两个指标不应混为一谈。

验收结果：

- CPU 数值、课程及四头 checkpoint 回归 **62 passed / 2 skipped**，后两项为 CPU 环境下跳过的 CUDA 参数；旧 NumPy 2 checkpoint 在 `pmt` 的 NumPy 1 环境中无法反序列化，相关回归改用已有 `isaacsim` 环境后通过，没有改写 checkpoint。
- `pmt` 环境、物理 GPU 1 的 CPU/CUDA 边界及课程测试 **56 passed**。覆盖 497/498/499/500 步、不同控制步上限、不同动作长度、部分 reset、重复读取、关闭容差，以及首次/自动 reset 的真实调用顺序。
- `pmt / Isaac Sim 5.1`、GPU 1、headless、5 环境有限验收 **passed**，覆盖五类十级、56 个控制步。第 498 步尚未到末帧时继续；随后第 499 步，或自动 reset 对应的第 498 步，末帧与 time_out 同时触发且课程成功升级。未耗尽动作在第 500 步超时，课程保持等级。**21 次 done 对应 21 次 sampler outcome**，没有重复结算；原四项 0.2 s 容忍、地图越界和接触缓存检查通过。
- [Isaac 验收记录](../logs/pgmt_timeout_boundary_20261015/isaac_01/training_status.json)、[原始边界记录](../logs/pgmt_timeout_boundary_20261015/isaac_01/measurements.json)。验收使用受控帧和计数探针，不代表策略自然完成 10 s 跟踪。进程已退出，没有启动 PPO 训练。

终止函数和参数会进入正式 `training_contract`。修改前的 checkpoint 与新配置不再是字面相同的严格续训契约，旧文件未修改，检查也未放宽；后续继续使用旧 checkpoint 时需要显式处理这项配置迁移。
