# PGMT P3.4：四头价值、正式地形任务与迁移/续训验收

日期：2026-10-14。**已完成实现与有限训练验收。** 新任务 `pgmt_stage2_terrain` 使用 `[upper, lower, terrain, auxiliary]` 四维 critic，接入 27 项奖励及三项下肢地形容差。160 环境真实 Isaac 完成 3 次更新、新进程恢复后 2 次更新、再次恢复检查，以及最终诊断代码下的 1 次续训更新；累计 6 次更新。415 项 CPU/CUDA 回归通过。所有本次 Isaac 进程均已退出。

**同日后续变更：** 用户随后修改了 [Stage2 终止条件](PGMT_STAGE2_TERMINATIONS.zh-CN.md)，包括 10 s 超时。本文运行结果和 checkpoint 保留为修改前的历史证据；这些 checkpoint 不能在新终止配置下按同配置严格续训。

**最新配置补验：** 双腕 EE acceleration、四项 0.2 s 连续容忍和 10 s 回合已在新的 160 环境 3+2 次更新、二次恢复和 P2 迁移中通过，600 项 CPU/CUDA 回归通过。该轮结果和验收 checkpoint 见 [P3 总验收](PGMT_P3_ACCEPTANCE.zh-CN.md)，不覆盖本文首轮历史记录。P3.5/P3.6 仍未全部完成。后续 P4 已加入 PhysX 等契约字段，当前入口的可续训 checkpoint 及 8192 运行结果见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。

这次验收证明迁移、训练和恢复链路可用；不代表地形跟踪性能已收敛或整个 P3/P4 已完成。恢复池及 recovery upward velocity 按用户要求暂缓；当时未加入的终止延迟现已按用户要求设为 0.2 s，无额外 reset 宽限，安全身体集合动态审计和分级评测仍待完成。

## 1. 正式任务与接口

| 接口 | 当前设置 |
| --- | --- |
| 任务 / 注册名 | `pgmt_stage2_terrain` / `PMT-PGMT-Stage2-Terrain-G1-v0` |
| 训练入口 | `bash scripts/train_pgmt_terrain.sh`，通过专用 runner 执行显式初始化或严格续训 |
| critic | 输入仍为 251 维特权 prop + 128 维 IFM intent；共享 MLP 的末层由 `128→3` 扩为 `128→4`，不是四个独立网络 |
| 输出和 reward 分组 | 严格按 `[upper, lower, terrain, auxiliary]`；`evaluate()` 与 `extras["reward_groups"]` 均为 `[N,4]` |
| 奖励数 | `6 upper + 6 lower + 6 terrain + 9 auxiliary = 27` |
| terrain 权重 | touchdown `+10`，reference contact match `+1.5`，slip `−1`，stumble `−20`，contact switching `−30`，contact force `−1e−6` |
| 下肢容差 | 默认开启；位置/姿态/关节位置上限 `0.20 m / 0.70 rad / 0.50 rad`，stairs/slope/boxes，当前 torso 所在 tile 的 `L/9`，逐部位 α 默认 1 |
| 数据和地形 | 862 个 clipped segment，动作级 adaptive、兼容表过滤、固定第 0 帧；v6 五类十级，随机出生 XY/yaw，出生地形课程独立记账 |
| 感知 | 既有 21×21 地图、PMT 增强与单次预处理、四个 glimpse、13 个 K/V 及 key mask；联合更新 selector/patch encoder/IFM/actor/critic |
| PPO | rollout 24，epochs 5，minibatches 4，γ=0.99，λ=0.95，初始 LR=1e−4、adaptive schedule |

保留旧 `pgmt_stage2` / `train_pgmt_stage2.sh` 作为 P2 感知基线：默认仍为 21 项奖励、三头、容差关闭。Stage1 默认网络和旧三头 metadata 不变。四头任务的入口使用新 wrapper，避免误启动旧 P2。

## 2. 三头到四头迁移

`--initialize-from` 支持指定 Stage1 的 `model_15000.pt`，也支持已经训练的 P2 三头感知 checkpoint。迁移首先校验动作、观测 schema/hash、组顺序、网络结构、感知契约和全部参数的形状/类型/有限性，再修改目标参数。

| 新行 | 语义 | 初始化 |
| ---: | --- | --- |
| 0 | upper | 旧行 0 的 weight/bias |
| 1 | lower | 旧行 1 的 weight/bias |
| 2 | terrain | weight 和 bias 全零 |
| 3 | auxiliary | 旧行 2 的 weight/bias |

其余旧主干、actor、critic 隐藏层、探索标准差和归一化状态逐张量精确复制。Stage1 新增的感知分支正常初始化；来自 P2 时已有感知参数也精确复制。迁移器按末个 Linear 定位输出层，当前结构对应 `critic.6`。

**迁移是新阶段初始化**：Adam 不继承三头状态，iteration 从 0 开始，sampler/地形课程重新初始化。原 checkpoint 文件不会被改写，来源路径、SHA256、组映射及复制/新建参数名单写入 `migration.json` 和新 checkpoint 的 `initialization`。

迁移一致性比较区分来源：Stage1 在禁用 terrain attention 时比较；P2 保持相同感知模式比较。本次两种真实来源的 actor 与三个保留价值输出最大误差均为 **0.0**，terrain 初值均为 0。Stage1 复制 47 个完整张量、重排行权重/偏置 2 个、新建感知张量 10 个；P2 复制 57 个完整张量、重排 2 个。迁移后正式训练开启感知，因此不要求训练动作仍等于盲走 Stage1。

CPU 测试中，3 行和 4 行矩阵运算可能采用不同的 float32 归约路径；因此参数要求逐位一致，旧价值输出使用 `atol=2e−5, rtol=2e−6`，另以 float64 检查复制行的线性计算。不能以完全相同的浮点输出代替参数检查，也不能因正常舍入差异放宽参数迁移规则。

## 3. 四头 PPO 的计算与日志

RewardManager 将每项原始率乘权重和 Δt 各一次；wrapper 将已积分分项按组相加。PPO 直接保存这四组奖励，不再次乘权重或 Δt，也不把标量总奖励复制成四份。

对每个价值头 h 独立计算 GAE 和 return：

$$
\delta_t^h=r_t^h+\gamma(1-d_t)V_{t+1}^h-V_t^h,\qquad
A_t^h=\delta_t^h+\gamma\lambda(1-d_t)A_{t+1}^h,\qquad
R_t^h=A_t^h+V_t^h.
$$

Actor 使用 `A_t = Σ_h A_t^h`，随后按原生 rollout/minibatch 开关归一化一次。保持四组相对权重，不分别标准化四组。critic 使用原生 clipped value loss，对样本和四头一起取均值；每头 loss 单独记录，但不额外加到优化目标中。

新增 TensorBoard / `progress.jsonl` 键：`value_function_upper/lower/terrain/auxiliary`，四者均值等于总 `value_function`。记录每头参数变化和跨本次所有 minibatch 的梯度 L1 累计；clipped value loss 可使某个头在最后一个 minibatch 梯度为零，因此不能仅凭最后一次 backward 判定断梯度。

**timeout 沿用原生语义**：每头在 timeout 奖励中加 `γ × transition.values`，使用采样动作时的旧状态 value，然后 done 切断 GAE；此次未改为截断前最终观测 bootstrap，更不会用 reset 后的新观测冒充旧回合末状态。四头 done/timeout 数值与广播由专项测试覆盖；本次短训没有自然发生 timeout，不能声称真实 rollout 覆盖了这一事件。

## 4. 严格续训

四头 checkpoint 使用 `kind=pgmt_terrain_training, version=1` 和四头 metadata。`--resume` 只接受同任务契约，恢复模型/归一化、Adam、动态 LR、下一迭代编号、累计步数/时间、地形课程/统计、动作 sampler 统计和待更新状态，以及 Python/NumPy/Torch CPU/CUDA RNG。

采样概率由恢复后的统计重新计算，实测与保存统计对应的概率逐值一致。RNG 在新的物理 reset 前恢复；不保存 PhysX 中断状态、不复演中断轨迹，重新初始化物理回合也不伪造 sampler/课程 outcome。

奖励权重、组顺序、容差配置、地形/动作身份、观测/感知契约、优化器结构等不一致时拒绝严格续训。三头 P2 checkpoint 必须走显式迁移；不能当成四头断点直接 `--resume`。反向四头转三头迁移未提供。

## 5. 验收结果与边界

机器：RTX 5090，`cuda:0`，Isaac Sim 6 compatibility / Isaac Lab 2.3，160 环境，headless；使用完整动作库、原生终止规则和实际 27 项奖励，没有替换成测试用奖励。

| 运行目录（`logs/pgmt_p3_four_head/`） | 过程 | 结果 |
| --- | --- | --- |
| `stage1_160_01` | 指定 Stage1 迁移，3 次更新 | next iteration=3；72 控制步，10 个 done，terrain 非零奖励 11,446 样本 |
| `resume_160_01` | 新进程从上述 final 恢复，2 次更新 | next iteration=5；48 步，3 个 done，terrain 非零奖励 7,603 样本 |
| `restore_160_01` | 新进程二次恢复，0 次更新 | next iteration=5；initial/final 的持久训练字段完全一致 |
| `p2_160_01` | 已有 P2 `logs/pgmt_p245/resume_160/final.pt` 显式迁移 | 0 次更新，验证已训练感知参数保留、四头结构及旧输出一致 |
| `diagnostics_160_01` | 从 restore 的 final 再恢复，最终诊断代码下更新 1 次 | next iteration=6；24 步，1 个 done，验证分头 loss 和累计梯度 |

合计 144 控制步 ×160 环境 = **23,040 transitions**、14 次自然失败结算。每一步四组和与标量奖励一致，最大绝对误差 **1.78813934e−7**；四个价值头均有非零梯度和参数变化，感知、IFM 与 actor 也发生更新。短训未出现地形升级，不能以该结果证明策略完成率。

最后一次更新的 value loss 为：upper `0.1413711`、lower `0.1833400`、terrain `0.7950681`、auxiliary `0.1350185`，总均值 `0.3136994`。四头累计梯度 L1 为 `37.50 / 55.95 / 586.98 / 58.47`。这些是梯度与日志验收数据，不用于宣称四组学习均衡或超参数已优化。

- **415 passed**：真实 Stage1/P2 checkpoint 的 CPU/CUDA 迁移、错误输入在修改前拒绝、元数据重建、四头 GAE/return/done/timeout/value loss、PPO 梯度、两轮严格恢复，以及既有感知/奖励/容差/课程回归。输出见 [cpu_cuda_tests.txt](../logs/pgmt_p3_four_head/cpu_cuda_tests.txt)。
- 独立读取保存文件复核迁移参数、来源哈希、Adam、环境状态及连续恢复，结果见 [acceptance_report.json](../logs/pgmt_p3_four_head/acceptance_report.json)。
- 各运行目录保存 `params/`、`launch.json`、`training_status.json`、`initial.pt`、`final.pt`、训练日志和感知调试结果。最新验收 checkpoint 为 [diagnostics_160_01/final.pt](../logs/pgmt_p3_four_head/diagnostics_160_01/final.pt)，它仅训练 6 次更新。

验收过程中修正了两个测试口径：避免将 float32 GEMM 舍入误判为迁移错误；避免将 clipped loss 最后一个 minibatch 的零梯度误判为某头未训练。用于梯度测试的观测按真实 checkpoint 的归一化统计构造，避免用任意单位尺度输入把 selector 推入饱和区。训练数值规则未因这些测试调整而改动。

首轮未扩容至 1024/8192 环境，也未重新验证 GUI 或论文的连续 30 s 完成率。当时待设计的终止延迟现已完成 0.2 s 版本及独立验收；P2 已定位的 PhysX 大规模初始化显存问题仍需正式接入修复。

## 6. 命令与代码位置

以下从 PMT 项目根目录运行。四头新训练示例（末行的迭代预算按实际需要设置）：

```bash
bash scripts/train_pgmt_terrain.sh --headless --device cuda:0 --num_envs 160 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir logs/pgmt_stage2_terrain/run_01 --max_iterations 20000
```

以下是**旧终止配置下的历史续训命令**（`max_iterations` 表示追加次数）。当前代码已启用新终止规则，直接执行会因契约变化被拒绝；新训练使用上面的 Stage1 初始化命令，新配置续训改用新运行生成的 checkpoint。

```bash
bash scripts/train_pgmt_terrain.sh --headless --device cuda:0 --num_envs 160 \
  --resume logs/pgmt_p3_four_head/diagnostics_160_01/final.pt \
  --log_dir logs/pgmt_stage2_terrain/resume_01 --max_iterations 1000
```

wrapper 默认启用 GUI；移除 `--headless` 可使用已有可视化界面。此次验收实际采用 headless，未额外启动长期训练。不要复用已有运行目录，入口会拒绝覆盖。

```bash
# 用当前配置重新执行迁移、3+2 次更新、二次恢复和 P2 迁移；另存新报告。
bash scripts/validate_pgmt_four_head.sh logs/pgmt_p3_four_head_next

# 复核本次已保存结果（包含最后补验的 1 次更新）。
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/report_pgmt_four_head.py \
  logs/pgmt_p3_four_head --diagnostics-run diagnostics_160_01
```

读取原生 checkpoint 使用 `isaacsim` Python 环境；文件含 NumPy 2 RNG 状态，旧 NumPy 1 的 `pmt` 环境可能报 `numpy._core` 缺失。这是读取环境版本差异，不应通过删掉 RNG 状态绕过。

| 代码 | 职责 |
| --- | --- |
| [pgmt_value_contract.py](../motion_tracking_rl/pgmt_value_contract.py)、[pgmt_actor_critic.py](../motion_tracking_rl/networks/pgmt_actor_critic.py) | 固定语义顺序、动态末层维数和 checkpoint metadata |
| [pgmt_stage2_terrain.yaml](../configs/task/pgmt_stage2_terrain.yaml)、[reward 配置](../configs/reward/pgmt_stage2_terrain.yaml)、[terrain_training.py](../pmt_tasks/env_cfgs/pgmt/terrain_training.py) | 正式四头任务、27 项奖励、默认容差 |
| [builder.py](../pmt_tasks/builder.py)、[agent_cfgs/pgmt.py](../pmt_tasks/agent_cfgs/pgmt.py)、[registry_gym.py](../pmt_tasks/registry_gym.py) | 环境/策略构建、组顺序校验和任务注册 |
| [pgmt_checkpoint.py](../motion_tracking_rl/utils/pgmt_checkpoint.py)、[pgmt_runner.py](../motion_tracking_rl/runners/pgmt_runner.py) | 显式三头迁移、新阶段初始化、严格四头保存恢复 |
| [ppo.py](../motion_tracking_rl/algorithms/ppo.py)、[rollout_storage.py](../motion_tracking_rl/storage/rollout_storage.py) | 按组奖励、四头 GAE/value loss、actor 优势求和和分头日志 |
| [train_pgmt_stage2.py](../scripts/train_pgmt_stage2.py)、[train_pgmt_terrain.sh](../scripts/train_pgmt_terrain.sh) | 实际训练入口、逐步分组/dt 核对、迁移与恢复检查、梯度/参数诊断 |
| [test_pgmt_four_head.py](../tests/test_pgmt_four_head.py)、[test_pgmt_rl.py](../tests/test_pgmt_rl.py)、[report_pgmt_four_head.py](../scripts/report_pgmt_four_head.py) | CPU/CUDA 与保存文件独立复核 |
