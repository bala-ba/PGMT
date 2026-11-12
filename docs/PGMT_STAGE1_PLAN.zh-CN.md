# 在 PMT 中训练 PGMT stage 1：实施方案

日期：2026-09-28。状态：已完成代码接入及短训练验证。实现、运行命令和实际验证结果见 [PGMT_STAGE1.zh-CN.md](PGMT_STAGE1.zh-CN.md)。用户限定本轮只做代码接入和短验证，正式长训练不在本轮范围内。

## 1. 已确认的选择

| 选项 | 确认结果 |
| --- | --- |
| 1A | PGMT 三头 critic；局部扩展 PMT storage/GAE；三组 advantage 求和后按 PMT 原生方式归一化 |
| 2B | 基准任务 `PMT-G1-MultiMotionV2-Flat-v0`，对应 `configs/task/multimotionv2_flat.yaml` |
| 3A | 使用 `../PGMT/assets/motions_50hz` 的全部 40 段 LAFAN1 G1 重定向动作 |
| 4A | 网络结构、输入定义和奖励公式以当前 PGMT 本地实现为依据 |
| 5A | 保留 PMT 原生重置，禁用依赖倒地池模式的恢复向上速度奖励 |
| 6A | 使用 PMT 的 CPU 独立 MuJoCo 路径进行闭环验证和可视化 |

主工程是 PMT。PGMT 提供网络和奖励的参考实现，以及重定向数据。训练执行、仿真和产物管理使用 PMT 原生流程。

## 2. 原生流程与新增范围

最终训练链：

```text
PMT/scripts/train.py
  → task YAML / builder / Gym registry
  → Isaac Lab ManagerBasedRLEnv + PMT G1
  → PMT RslRlVecEnvWrapper
  → PMT OnPolicyRunner
  → PMT PPO（增加三价值头支持）
  → PMT RolloutStorage（增加三组 returns，保留标量 actor advantage）
```

继承 2B 的设置：

- `MultiMotionCommandV2`、`bin_adaptive`、原生采样统计和更新机制。
- eager 动作加载、FP32 数据、4000 帧 chunk；控制频率 50 Hz，物理步长 0.005 s、decimation 4。
- PMT G1 资产、执行器、PD 参数、关节限位、物理材质和随机化。
- 动作：`q_target = q_default + G1_ACTION_SCALE * action`。
- 原生 reset、reference 推进、termination 和回合时限。
- PMT 观测归一化模块及更新时机、Gaussian 分布和探索配置。
- PMT PPO 优化循环、KL 自适应学习率、梯度裁剪、minibatch 规则和 timeout bootstrap 约定。
- 原生训练入口、日志、模型保存、加载与恢复语义。

新增范围为 PGMT 网络、对应观测、PGMT 奖励、LAFAN1 格式适配、三价值头的最小 PPO 支持，以及 CPU MuJoCo 的新策略接线。PGMT 独立环境、训练 runner、RunningNorm、PPO 调度、恢复池和 campaign 不进入该任务。

2B 的原生 PPO 起点为 rollout 24、epochs 5、minibatches 4、初始学习率 1e-3、gamma 0.99、lambda 0.95、clip 0.2、entropy 0.005、desired KL 0.01、max grad norm 1.0、初始动作标准差 1.0。实际值以 builder 生成的 agent config 为准，保存完整解析结果。

## 3. PGMT 网络和观测

以 `../PGMT/pgmt/modules.py`、`../PGMT/pgmt/environment.py` 和 `../PGMT/configs/optimized_v2.yaml` 为结构及输入依据，在 PMT 中注册 `PGMTActorCritic`。

- 当前 proprioception：96 维。
- 历史：10 × 96，以当前状态查询历史的 cross-attention。
- 未来命令：8 × 61；偏移为 0、1、3、7、15、31、63、127 帧。
- intent fusion：RoPE cross-attention。
- embedding 128、4 个 attention heads；actor/critic hidden widths 为 512、256、128。
- actor 输出 29 维；critic 输出顺序固定为 `upper, lower, auxiliary`。
- stage 1 使用平地，不引入地形 token。

输入通过 PMT ObservationManager 构建；保留 PGMT 的输入字段含义及局部坐标计算。姿态/速度缩放、历史时序及 future command 必须有明确 layout。PMT 原生 286 维输入不能直接传入新网络。

需要特别处理：

1. PGMT 当前 proprioception 的末 29 维实际是上一控制步的目标关节角。适配层读取 PMT 实际执行的目标角，不能用 raw action 替代。
2. 动作解码使用 PMT 的 `G1_ACTION_SCALE`；PGMT 独立环境中的 action clip、scale 重算规则不移植。
3. reference 的推进和 chunk 边界遵循 PMT。未来窗口在 chunk 尾部的处理使用 PMT 数据访问约定，并单独验证。
4. 网络暴露 PMT 所需的 `act/evaluate/act_inference/get_actions_log_prob`、分布属性和归一化接口；历史由环境提供，`is_recurrent=False`。
5. 归一化模块、统计更新时机和探索配置按 PMT 实现。网络结构复用不意味着整体复制 PGMT 的 `Policy` 包装器。

## 4. PGMT reward 接入 RewardManager

以当前 `../PGMT/pgmt/environment.py:rewards` 的实际公式和 `optimized_v2.yaml` 的误差尺度为准。保留 upper、lower、auxiliary 三组；stage 1 的 terrain relaxation 为零。

在 PMT 中新增 reward terms 和配置，覆盖原生 DeepMimic reward 集合。使用名称映射获取 PGMT 公式需要的刚体/关节集合；不能直接用 PMT 默认 tracked-body 子集替代。

关键约定：

- `recovery_upward_velocity` 明确设为禁用并记录原因；不以低身高或动作文件名替代 recovery mode。
- `action_rate` 计算实际目标关节角变化，保持当前 PGMT 公式含义。
- 身体误差、纠偏根速度和 floating-anchor 项使用对应 PGMT 坐标及公式。
- 加速度和动作变化项的历史缓存按控制步更新，重置时清理；反复读取观测或奖励不能推进缓存。
- PMT command 在动作片段结束时可能直接重采样和重置机器人，没有对应的环境 done。新增的 PGMT 历史及奖励缓存要同步处理这种边界；保持 PMT 原生 episode mask，不自行把它改成 PGMT 的 truncation。
- Isaac RewardManager 负责时间积分，移植的奖励函数不再额外乘 dt。
- 同时记录每项奖励和三组加权贡献，核对三组之和等于环境总奖励。
- 奖励读取 PMT 当步的 reference 状态，reference 推进和 reset 顺序继续由 PMT 管理。

## 5. 三价值头的最小 PPO 扩展

环境继续向原生 runner 返回标量总奖励 `[N]`。通过 `extras["reward_groups"]` 提供三组当步贡献 `[N, 3]`，供 PPO 写入 storage；这样 runner 的 episode reward 累计和日志接口保持原样。

| 数据 | 形状 |
| --- | --- |
| 环境总 reward | `[N]` |
| 分组 reward / value / return | `[T, N, 3]` |
| actor advantage / log probability | `[T, N, 1]` |
| action / mean / std | `[T, N, 29]` |

每组独立按 PMT 的 gamma、lambda 和 episode mask 计算 GAE：

```text
A_sum = A_upper + A_lower + A_auxiliary
A_actor = PMT_normalize(A_sum)
```

使用 PMT 原生标准差估计、epsilon 和归一化范围；不执行各组单独标准化。critic clipped value loss 在样本和三个输出上取平均，value-loss coefficient 沿用 PMT。

保留 PMT timeout bootstrap 的既有约定，仅扩展价值维度；此前考虑过的 PGMT `final_obs` bootstrap 不属于本次已确认范围。归一化时机、KL 规则及优化器更新也不改成 PGMT 的版本。

实现以默认单头行为兼容为要求：旧任务仍走原有单头分支；新增多头配置必须在缺少或错配 reward groups 时明确报错。reward groups 必须在 reset 前保存，避免回合结束后读到已清空统计。

## 6. LAFAN1 数据转换

将源数据转换到 PMT 独立的数据目录，保留全部 40 段和原始文件，不进行训练/测试子集替换。

```text
body_pos     → body_pos_w
body_quat    → body_quat_w
body_lin_vel → body_lin_vel_w
body_ang_vel → body_ang_vel_w
fps 标量     → shape (1,)
```

校验 29 个关节、30 个刚体的名称与顺序、wxyz 四元数、50 Hz、帧数、有限数值和 PMT G1 的 FK 一致性。静态检查已确认相关 URDF、预处理 XML 和名称顺序一致；正式转换仍需生成机器可读校验报告。

记录输入输出路径、哈希、帧数、名称映射和转换版本。预期总量为 441,140 帧，以实际读取核对结果为准。

若评测使用逐帧 contact 元数据，应按原 clip/frame 或随 chunk 切分的数据访问，不能用 chunk 内索引直接读取整条 contact metadata。stage 1 不启用 stage 2 的 terrain-contact reward。

## 7. Checkpoint 与 CPU MuJoCo

### Checkpoint

使用 PMT 的 `model_state_dict / optimizer_state_dict / iter / policy_metadata` 格式。metadata 描述 PGMT 网络结构、观测 layout、价值组顺序和动作约定。PMT 及 PGMT 参考代码均记录实际文件哈希，覆盖工作区未提交改动。

模型从 PMT 的 scratch 流程训练。当前 PGMT checkpoint 不默认作为初始化或续训来源。恢复范围遵循 PMT 原生实现，不承诺物理状态逐位恢复；2B 的 sampler 统计也不能在未验证保存逻辑前宣称已恢复。

### CPU MuJoCo

沿用 `scripts/export_mujoco_contract.py`、`pmt_tasks/backends/mujoco_standalone.py` 和现有评测/可视化入口，为新 task 增加 contract 和策略分发。

现有 standalone 针对 Bigmap vision student，不能直接套用其 `raw_q_ref + action` 解码或配对动作要求。新增 PGMT flat 分支需要：

- 从实际 PMT 训练环境导出 G1 控制参数、joint/body order、默认角、action scale、观测与参考命令定义。
- 加载 PMT 格式的 PGMT checkpoint 和归一化状态。
- 重建 96 维输入、10 帧历史和 8 帧指数间隔命令；执行 `q_default + scale * action`。
- 使用 PMT 原生 G1 模型构建、物理步进和可视化路径。
- 先做同状态的观测/动作/目标关节角数值对齐，再进行真实 MuJoCo 闭环 rollout。

## 8. 预计文件改动

以下为实现采用的主要文件布局。

| 文件或模块 | 改动 |
| --- | --- |
| `motion_tracking_rl/networks/pgmt_actor_critic.py` | PGMT 网络和 PMT policy 接口 |
| `pmt_tasks/mdp/pgmt_observations.py`、`pgmt_rewards.py` | 观测、奖励和所需缓存 |
| `pmt_tasks/env_cfgs/pgmt/stage1.py` | 继承 2B，配置新观测和奖励 |
| `pmt_tasks/agent_cfgs/pgmt.py` | 继承 2B runner/PPO 参数，配置 PGMT network |
| `configs/task/pgmt_stage1.yaml` 及对应 network/obs/reward 配置 | 新任务组合 |
| `pmt_tasks/builder.py`、`registry_gym.py`、RL registry/compat/imports | 新 task 和 network 注册 |
| `motion_tracking_rl/algorithms/ppo.py`、`storage/rollout_storage.py` | 可选三组价值支持，默认单头行为兼容 |
| 新任务的 reward extras 接线 | 为 PPO 提供三组贡献，保持 runner 标量 reward 接口 |
| `scripts/prepare_pgmt_lafan.py` | 数据转换、清单和一致性校验 |
| CPU MuJoCo contract / policy loader / observation 分支 | 接入新网络和 flat 控制约定 |
| `tests/` | 数值、接口和原任务回归验证 |

## 9. 实施顺序与验收

1. **冻结基线与数据。** 保存 2B 解析后的配置和源码指纹；转换 40 段数据，核对帧数、顺序、FK 和 chunk 边界。
2. **网络与奖励。** 在固定状态、固定参考下验证 PGMT 网络计算和逐项奖励；保持不同归一化与控制包装的边界清晰。测试 reset 和重复读取不会污染缓存。
3. **PPO 三头接线。** 验证独立 GAE、advantage 先求和再标准化、三头 value loss、timeout/reset 和 reward extras；覆盖 command 内部换片但没有 done 的缓存边界；单头路径与原 PMT 做数值回归。
4. **小规模 Isaac 验证。** 建议 64–128 环境、20–50 次更新，验证梯度、有限数值、原生保存/加载、bin sampling 更新及原任务回归。规模是验证建议，不是正式训练预算。
5. **CPU MuJoCo。** 同状态比较观测、确定性动作和目标角，再对全部 40 段运行闭环；记录失败类型、跟踪误差、持续高度丢失和视频。评测诊断不自动改写 PMT 训练终止规则。
6. **正式 stage 1。** 后续由用户指定运行预算；本轮明确只完成代码接入和短训练验证。

验收报告区分接口数值正确、训练能够进行和跟踪能力三个结果。三头结构和 PGMT reward 接入完成，并不自动意味着达到论文性能。
