# Stage2：P-BFM 动作采样与 EMD 优先级

2026-11-11 起，`pgmt_stage2` 和 `pgmt_stage2_terrain` 默认使用
[`configs/motion/pgmt_emd.yaml`](../configs/motion/pgmt_emd.yaml)。
Stage1 和历史 P1 的采样保持原配置。

## 训练采样

1. 根据出生地形读取原有兼容表，排除禁配动作。
2. 在兼容动作集合中按 EMD 权重归一化，抽取一个完整 NPZ clip。
3. 在选中 clip 内均匀抽取起始时间；不使用帧级失败率或时间段权重。
4. 使用该帧的参考姿态、速度，在地块中心放置机器人并随机化朝向。

PMT 的参考系统按整数控制帧推进。因此时间采样等价于
`floor(U[0, (N-1)/fps) * fps)`。当前每条数据为 50 Hz / 499 帧，
训练起点均匀落在帧 0～497（0～9.94 秒），末帧 498 不单独作为起点，
保证至少有一个真实控制区间。不会在长动作中在线另裁一个 10 秒窗口。
`chunk_length=0`、`enable_frame_sampling=false`、`fixed_start_frame=-1`。

地形兼容过滤、真实碰撞放置和关节限位检查仍有效，足部支撑过滤关闭。
地形课程的成功条件相应变为：同一 clip 从采样起点执行完剩余参考，
并经过足够的真实控制步。跟踪失败、越界与其他截断仍按各自原因结算。
训练仍在参考结束时 reset；没有改为 P-BFM 的 motion-end 不终止行为。

## EMD 评估与权重

载入时所有 clip 等权；**开始训练或恢复训练时先评估一次**，然后用新权重
重新初始化训练 episode，不把中断的 episode 记作成功或失败。随后每跨过
**9,600,000 个全局环境 transition** 的周期边界，在最近的
PPO 更新边界暂停采集并运行一次完整动作库评估。全局步数为
`completed_iterations × num_steps_per_env × local_num_envs × world_size`。
评估默认每批 64 个环境，种子 1729。

评估使用独立 Isaac 进程中的平地、固定朝向、动作第 0 帧和确定性 actor mean，
关闭推力、随机动力学和观测噪声。它记录每条 clip 的完整关节角序列；
不因普通跟踪失败提前 reset，不把多个 episode 拼起来计算分数。
周期评估不改变训练中的物理 episode、训练 RNG、策略参数或归一化统计。
开始/恢复训练的首次评估后按新分布重置 episode，与 P-BFM 的启动流程一致。

与本地 P-BFM 一致，代价矩阵是每对机器人/参考帧之间的关节角欧氏距离，
所有帧质量均匀。两条序列都含 N 帧，使用 SciPy 精确线性指派的平均代价，
它与该条件下的 `ot.emd2` 等价；不使用对齐 RMSE、平方距离、Sinkhorn 或时间降采样。

```
weight[i] = 2 ** (2 * clamp(EMD[i], 0.5, 2.0))
P(i | terrain) = weight[i] / sum(weight[compatible clips])
```

权重范围为 2～16，最大比值为 8。每次完整评估后直接替换权重；没有 EMA、
失败率更新或额外的 10% 均匀混合。每个 clip 在全部地形间共享一个权重，
各地形只改变候选集合。当前数据库作为一个数据源使用。

多进程训练由 rank 0 评估完整动作库，并向全部 rank 广播同一组分数；评估失败
也会广播到所有 rank。缺失分数、非有限值、意外 reset 或 PhysX 错误使本次评估失败，
不会用部分结果覆盖已有权重。日志和每个动作的分数保存在训练目录的
`emd_evaluations/step_*/`，TensorBoard 使用 `EMD/*`。

## 启动与恢复

现有 Stage2 训练入口自动启用上述配置。评估运行时继承训练进程的 Python、
Isaac Lab 路径、Kit experience 和 device，无需另装 POT。

旧失败率采样 checkpoint 与新训练契约不同，不能直接 `--resume`。
可以通过 `--initialize-from <旧 checkpoint>` 仅加载匹配的策略及归一化权重，
重新开始优化器、地形课程和 EMD 权重。新的 checkpoint 保存 clip 权重、EMD 分数、
评估次数与最近评估的全局步数，严格续训会恢复这些状态；实际进入训练循环时
会按 P-BFM 的规则再次评估当前策略。

本次验证包含采样分布、精确 OT、原子更新/恢复和双进程广播测试，以及
[`scripts/validate_pgmt_emd.py`](../scripts/validate_pgmt_emd.py) 原生 Isaac 回归。
该回归覆盖随机出生、普通/快速放置、部分 reset、一次 PPO 更新、独立完整片段评估
以及严格续训；生产默认仍评估全部 862 个 clip。

验证结果：205 项相关 CPU 回归通过，包含双进程 Gloo 权重/异常广播；
原生 CPU 和 CUDA 均完成随机出生、PPO、EMD、恢复闭环。最终 CUDA 回归
记录在 [`logs/pgmt_emd_integration_final/result.json`](../logs/pgmt_emd_integration_final/result.json)，
包含 180 次出生检查、两个完整 499 帧片段，以及 step 0 / step 20 两轮评估。
这里的 20 步仅用于回归，生产配置仍为 9,600,000 步。

测试选择排除了 CUDA/NCCL 单元变体（CUDA 用原生闭环另验）和一个已有奖励权重断言：
`test_formal_task_has_27_rewards_four_groups_and_enabled_relaxation` 仍期望 T1 权重 10，
工作区现有配置是 1.5。本次没有更改该奖励配置。没有执行全库长期训练或收敛评测。
