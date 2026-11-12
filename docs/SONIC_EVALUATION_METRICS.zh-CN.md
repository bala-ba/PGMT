# SONIC 仿真评测：success 与 metrics 代码核对

核对日期：2026-09-29。研究目录：`/home/lzp/projects/GR00T-WholeBodyControl`，HEAD 为 `1983e88`。以下依据本地当前文件；`im_eval_callback.py` 有未提交的 body-group 配置改动，`tools/g1/batch_eval/` 为本地未跟踪代码。本文没有修改被研究仓库，也没有运行新的策略仿真。

## 1. 两条评测路径

| 路径 | 入口 | 主要输出 |
| --- | --- | --- |
| SONIC 训练框架评测，Isaac Lab | `gear_sonic/eval_agent_trl.py` → `ImEvalCallback` | `eval/success/*`、`eval/all/*`、逐动作字典 |
| 本地 G1 MuJoCo 批量评测 | `tools/g1/batch_eval/run.py` → `worker.py` → `metrics.py` | success/failure/test_error、轨迹误差、动力学及运行质量指标 |

`gear_sonic/utils/mujoco_sim/metric_utils.py` 只提供接触/高度查询，不能用它解释完整的动作评测成功率。

## 2. SONIC success 的定义

对第 i 条动作，长度为 N_i 个控制步，回调累计失败标志：

```text
died = done AND NOT time_out
failed_i |= died_i AND (curr_steps <= N_i - 1)
success_i = NOT failed_i
success_rate = sum(success_i) / 评测动作数量
```

成功是整条参考动作期间没有被累计为提前失败。MPJPE、奖励、绝对骨盆高度均不直接进入回调的 success 公式。失败标志锁存，失败后再恢复也不会重新变为成功。`time_out` 帧被排除；同一步同时出现 timeout 和失败时也会被该掩码排除。超出该动作长度后出现的终止不再影响其 success。

证据：[累计失败逻辑](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:363)、[最终 success/progress](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:692)。

### 2.1 `tracking/eval` 声明的失败条件

默认 G1 anchor 为 pelvis，条件逐控制步检查，单步超过阈值即可触发。

| 条件名 | 失败条件 | 单位 |
| --- | --- | --- |
| `anchor_pos` | 参考与机器人 pelvis 的 Z 高度绝对差 > 0.25 | m |
| `anchor_ori_full` | 完整四元数最短旋转角 θ 的平方 > 1.0，即 θ > 1 rad，约 57.3° | rad² |
| `ee_body_pos` | 两个 ankle-roll 与两个 wrist-yaw 中任意一个的 Z 高度误差 > 0.25 | m |

四元数角度可写为 `θ = 2 acos(clamp(abs(dot(q_ref, q_robot)), 0, 1))`，四元数先归一化。完整姿态包括 yaw。

`anchor_pos` 实际只检查 Z，`ee_body_pos` 实际只检查末端 Z；名称中的 pos 不能解释成完整 XYZ 距离。末端函数读取 `body_pos_relative_w`，其水平位置和 yaw 已对齐，但 Z 保留参考高度，所以等价于比较参考和实测末端的世界 Z。

这些条件使用相对参考误差：参考本来就在爬行或倒地时，低绝对高度并不自动失败。单看这三项，没有全局 XY 漂移阈值，也没有 MPJPE 阈值。

正常动作结束条件为 `motion.time_steps + motion_start_time_steps + 1 >= motion_num_frames`，并标记 `time_out=True`，不是固定 15 秒。

证据：[eval 配置](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/config/manager_env/terminations/tracking/eval.yaml:3)、[高度和姿态函数](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/terminations.py:97)、[末端高度及 timeout](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/terminations.py:206)、[参考对齐](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/commands.py:3228)。

### 2.2 必须看最终合并配置

`sonic_release.yaml`、`sonic_v1_1.yaml` 的训练默认不是 `tracking/eval`，而是 `tracking/base_adaptive_strict_ori_foot_xyz`。其中包含更严格的高度/姿态设置、低姿态自适应高度阈值和 `foot_pos_xyz`。

加载 checkpoint 时，`eval_agent_trl.py` 将训练 `config.yaml`、`eval_overrides` 和当前参数递归合并。`tracking/eval` 未声明 `foot_pos_xyz`，因此训练配置中的该项可能继续存在。用本地 Hydra/OmegaConf 验证，先有 `foot_pos_xyz.threshold=0.2`，再合并 `tracking/eval`，该项仍保留。若保留，它会额外检查任一足端对齐后的 XYZ 距离 > 0.2 m。

所以只有检查最终 resolved configuration，才能断言某次 SONIC 评测究竟用了哪些成功条件。不能仅凭命令包含 `+manager_env/terminations=tracking/eval` 就认定只有上述三项。

证据：[配置合并](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py:79)、[训练终止配置](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/config/manager_env/terminations/tracking/base_adaptive_strict_ori_foot_xyz.yaml:3)。

### 2.3 `progress_rate`

每步先更新失败标志，再给仍未失败的环境增加一个 progress step。失败动作按累计步数 / N_i 计算，首次失败帧不增加 progress；最终成功动作的 progress 强制置为 1。所有动作的 progress 做等权平均。

success 和 progress 均按动作等权；误差指标的总体统计按帧加权。

## 3. SONIC 主要误差公式

令 x[t,j] 为实测身体位置，r[t,j] 为参考位置，单位 m；T_i 为第 i 条动作实际传给误差函数的数组长度，J 为身体点数量。默认 G1 的 J=14：pelvis、torso，以及双侧 hip-roll、knee、ankle-roll、shoulder-roll、elbow、wrist-yaw。

这些是机器人刚体原点的位置，不能解释成 29 个关节的角度误差。标准主评测没有计算 29-DOF joint RMSE。

### 3.1 位置误差

对单条动作：

```text
MPJPE-G = 1000/(T*J) * Σ[t,j] ||x[t,j] - r[t,j]||₂

MPJPE-L = 1000/(T*J) * Σ[t,j]
          ||(x[t,j]-x[t,0]) - (r[t,j]-r[t,0])||₂
```

- `mpjpe_g`：世界坐标位置误差，单位 mm，包含全局路径偏移。
- `mpjpe_l`：各自减去第 0 个身体位置。全身默认第 0 个是 pelvis；只去平移，不去 yaw 或完整旋转，也不缩放。pelvis 自身的局部误差为 0，但仍计入 J。
- `mpjpe_pa`：对每帧两个点集拟合最优尺度、旋转和平移，再平均对齐后的距离，单位 mm。SVD 拟合限制旋转不含镜像。这是带尺度的相似变换对齐，低误差不保证真实路径和落脚位置正确。

主回调只向 `compute_metrics_lite` 传位置，没有传旋转，所以该调用不会产生可选的 `rot_error`。依赖库里的 `TTR`、`skate`、`penetration` 等其他函数结果也不属于这个调用的输出。

证据：[14 个身体名称](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/config/manager_env/commands/terms/motion.yaml:47)、[实际调用](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:602)、[误差函数](/home/lzp/.venvs/sonic-training/lib/python3.11/site-packages/smpl_sim/smpllib/smpl_eval.py:58)、[PA 拟合](/home/lzp/.venvs/sonic-training/lib/python3.11/site-packages/smpl_sim/smpllib/smpl_eval.py:98)。

### 3.2 `vel_dist` 和 `accel_dist`

先在世界坐标计算位置差分，尚未减去根部位置：

```text
Δx[t]  = x[t+1] - x[t]
Δ²x[t] = x[t+2] - 2*x[t+1] + x[t]

v_error[t] = 1000/J * Σ[j] ||Δx[t,j] - Δr[t,j]||₂
a_error[t] = 1000/J * Σ[j] ||Δ²x[t,j] - Δ²r[t,j]||₂
```

实现没有除以 dt 或 dt²，所以是毫米尺度的一阶/二阶帧差分误差，不能标成 m/s、m/s² 或 jerk。50 Hz 时，单个一阶差分值乘 0.05 才对应 m/s，二阶差分值乘 2.5 才对应 m/s²。

汇总还有一个细节：速度只有 T−1 个差分、加速度只有 T−2 个差分，但 callback 将两者求和后仍除以 T。因此换算总体输出时，除了单位，还要修正分母；多动作时分别涉及总 T、总 (T−1) 和总 (T−2)。

数值验证：6 帧中有恒定 1 m/s 的轨迹速度差时，原始 `vel_dist` 差分值为 20，但 callback 输出 100/6=16.6667。恒定 2 m/s² 加速度差时，差分值为 0.8，callback 输出 3.2/6=0.53333。

证据：[一阶/二阶差分](/home/lzp/.venvs/sonic-training/lib/python3.11/site-packages/smpl_sim/smpllib/smpl_eval.py:295)、[共同的 T 分母](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:629)。

### 3.3 各身体区域的 metrics

回调先截取身体子集，再调用同一个 `compute_metrics_lite(root_idx=0)`。因此区域 MPJPE-L 使用该子集第一个身体作平移基准：

| 后缀 | 点集 | 局部平移基准 |
| --- | --- | --- |
| `_legs` | 双侧髋、膝、踝，6 点 | left_hip_roll_link |
| `_vr_3points` | torso、左腕、右腕，3 点 | torso_link |
| `_other_upper_bodies` | pelvis、双肩、双肘，5 点 | pelvis |
| `_foot` | 左踝、右踝，2 点 | left_ankle_roll_link |

例如 `mpjpe_l_foot` 中左脚误差恒为 0，右脚计算两脚相对位置的误差，再除以 2。它不能解释成两脚相对骨盆或世界坐标的平均落脚误差。区域的 PA 也单独拟合，两点足部 PA 尤其不能充当严格跟踪质量标准。配置可通过 `eval_body_groups` 覆盖上述点集及顺序。

证据：[分组与切片](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:547)。

## 4. 汇总及当前实现的两个问题

### 4.1 汇总规则

每条动作先产生逐帧误差；MPJPE 对身体取平均、对时间求和。最终：

```text
eval/all/M     = 全部动作的 M 误差和 / 全部动作的 T 之和
eval/success/M = 成功动作的 M 误差和 / 成功动作的 T 之和
```

长动作权重较大。`all_metrics_dict` 保留逐动作的误差和/T、terminated、progress、motion_keys、sampling_prob；后者是采样概率，并不是 success 的权重。

若没有任何成功动作，当前 `success_metrics_mean` 路径会出现 0/0，没有专门的零样本保护，因此可能为 NaN。

有物体的任务另记录 `obj_pos_error`（位置距离，m）和 `obj_ori_error`（完整姿态角，rad）。先对每条动作时间平均，再按动作等权平均，与身体误差的帧加权口径不同。普通 G1 动作跟踪不一定产生这些项。

证据：[总体/成功子集汇总](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:704)、[物体误差采集](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:304)。

### 4.2 失败动作没有按首次失败截断误差序列

回调每步采集所有环境的位置。批次结束后按 `[:N_i-1]` 切片，并没有用首次失败帧或 `progress_state` 作为切片上限。所以实际 T_i 是 `min(批次收集步数, N_i−1)`，不等于首次失败前的步数。

当前 ManagerEnvWrapper 路径调用标准 ManagerBasedRLEnv，其 `step` 会自动重置失败环境，再返回到回调。评测模式的 command 重置回该环境对应动作的第 0 帧。因此，失败后的重置/重跑状态可能继续进入 `eval/all/*`，甚至带来重置跳变造成的差分误差。success 仍由锁存的首次失败标志决定，但误差不能严谨地称为“只统计首次失败前有效前缀”。

例：两个环境都对应 100 步动作，一个在第 5 步失败，另一个继续到 100 步；当前切片会给前者也保留 99 帧，并非 4 或 5 帧。

证据：[采样与切片](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:340)、[实际 [:N−1] 切片](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py:449)、[自动 reset 后返回](/home/lzp/projects/IsaacLab-2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:215)、[评测 reset 起始帧](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/commands.py:2884)。

### 4.3 参考/实测数据命名对调，影响 PA 拟合方向

`get_env_data("ref_body_pos_extend")` 返回 `robot_body_pos_w`；`get_env_data("rigid_body_pos_extend")` 返回参考 `body_pos_w`。回调据名字构造 gt/pred 后，在此路径中实际是将参考传给 predicted、实测传给 target。

G/L 距离和差分误差对两者交换是对称的，不受影响。但带尺度的 PA 拟合不是交换对称：当前方向为“参考点集拟合到实测点集”。数值小例子确认两方向可分别得到 194.72 mm 和 185.53 mm，因此解释/复现 PA 时必须保留真实调用方向。

证据：[位置接口](/home/lzp/projects/GR00T-WholeBodyControl/gear_sonic/envs/wrapper/manager_env_wrapper.py:985)、[PA 拟合尺度](/home/lzp/.venvs/sonic-training/lib/python3.11/site-packages/smpl_sim/smpllib/smpl_eval.py:128)。

## 5. 本地 G1 MuJoCo batch_eval

### 5.1 success、failure、test_error

从参考第 0 帧开始，直到首次失败或最后一帧结束。硬编码阈值为骨盆高度 0.25 m、完整骨盆角误差平方 1 rad²、四末端高度 0.25 m，并额外把非有限状态或 BADQPOS/BADQVEL/BADQACC 视为物理发散。

没有失败并到达末帧为 success；有失败原因为 failure；进程/初始化异常、缺帧、时钟不同步或未完整运行归为 test_error。若末帧同时触发失败，本路径判 failure，与 Isaac 回调的 timeout 掩码需要区分。

```text
success_rate = 成功条数 / (成功条数 + 失败条数)
```

test_error 不进入分母；只有全部选定动作都有有效结果时才报告针对完整集合的最终成功率。另给 Wilson 95% 二项比例置信区间。

`completion_rate`：成功为 1；失败时为首次失败帧号 f/N（帧从 0 编号）。轨迹 metrics 包含第 f 帧，但 progress 计入的是失败之前的帧。总体 `mean_completion` 按有效动作等权平均。

证据：[硬编码失败条件](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/metrics.py:15)、[首失败结束](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/worker.py:112)、[结果分类和完成率](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/worker.py:162)、[成功率分母](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/run.py:37)。

### 5.2 MuJoCo 各项指标

参考只做一次初始 heading 旋转和固定 XY 平移对齐，随后不逐帧纠正漂移。所有指标只用实际记录的首次失败前缀，包含第一帧失败状态；失败后不重置继续计分。

| 字段 | 计算 |
| --- | --- |
| `mpjpe_g_mm` | 14 个身体世界位置距离，身体平均，再按帧统计，×1000 |
| `mpjpe_l_mm` | 双方分别减自己的 pelvis，14 点距离平均，×1000；不去朝向 |
| `root_position_mm` | pelvis 的 3D 距离 ×1000 |
| `root_xy_mm` | pelvis 的 XY 距离 ×1000 |
| `root_orientation_rad` | 完整四元数最短旋转角 θ |
| `feet_local_mm` | 两踝相对各自 pelvis 的距离误差平均，×1000 |
| `hands_local_mm` | 两腕相对各自 pelvis 的距离误差平均，×1000 |
| `joint_mse_rad2` | 每帧 29 关节角差平方的平均，不做角度 wrapping |
| `joint_rmse_rad` | 全部有效帧和关节的 MSE 平均后开根，不能先逐帧开根再平均 |
| `body_velocity_error_m_s` | 世界位置误差的一阶差分 / 0.02，14 点距离平均 |
| `body_acceleration_error_m_s2` | 世界位置误差的二阶差分 / 0.02²，14 点距离平均 |
| `peak_joint_speed_rad_s` | 1000 Hz 物理步中 29 个关节的最大绝对角速度 |
| `peak_joint_torque_nm` | 1000 Hz 物理步中 29 个关节的最大绝对实际力矩 |
| `torque_saturation_ratio` | 未裁剪 PD 力矩达到限幅的关节×物理步数量 / 全部关节×物理步数量，比较容差 1e-6 |

速度/加速度只用连续参考帧的相邻对/三元组，不跨缺帧差分；分母使用各自实际差分样本数。它们有 SI 单位，与 SONIC 训练评测的 `vel_dist`/`accel_dist` 不同。实时 DDS 非同步步进时仍使用名义参考间隔 0.02 秒，因此采样抖动会影响结果。

每条动作的标量逐帧指标保存 mean、p95、sum、count；joint RMSE 为单独标量。总体同时保存：

- `clip_mean`：逐动作均值再等权平均。
- `frame_weighted_mean`：各动作 sum 相加 / count 相加。
- 总体 joint RMSE 从汇总 MSE 开根。
- `all_valid_prefix_metrics`：所有有效动作，包括失败前缀。
- `success_only_metrics`：只取成功动作。

物理诊断总体最大值取各动作最大值的最大值；力矩饱和率重新汇总分子/分母，不平均动作饱和率。p95 是单动作逐帧分布的 95 分位，不是汇总置信区间。

运行质量项：`realtime_factor=Δsim_time/Δwall_time`；至少 5 帧时偏离 1 超过 2% 会被判为 test_error。还记录参考采样对应的实际物理时间间隔均值/最小/最大值和 wall lag。

证据：[轨迹与统计公式](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/metrics.py:28)、[物理诊断](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/worker.py:128)、[汇总函数](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/metrics.py:69)、[固定参考对齐](/home/lzp/projects/GR00T-WholeBodyControl/tools/g1/batch_eval/worker.py:94)。

## 6. 对 PMT/PGMT 对比的意义

此前 PMT 测试的 4.39 cm 是原生身体参考经过水平位置和 yaw 对齐后的误差，而 SONIC MPJPE-L 只减 pelvis 平移，两者不能直接比较。此前 MuJoCo 视频是选定时间线的 15 秒 rollout，不能给出完整动作库的 SONIC success rate。

若后续要统一评测，应明确：最终生效的终止项；14 个身体及根部定义；初始状态和随机化；整条动作从头执行；首失败即停止；失败帧是否计入；G/L/PA 的对齐及 PA 方向；差分单位和分母；按帧与按动作两种汇总；test_error 单独报告。

## 7. 本次验证

- 仅读取源码、配置和安装依赖。
- 本地 MuJoCo `tools/g1/batch_eval/test_metrics.py`：8 项测试全部通过。
- 从已安装 `smpl_eval.py` 提取原始纯函数执行数值例子，验证帧差分单位、T 分母效应和 PA 方向不对称。
- 用本地 Hydra/OmegaConf 合成并合并 eval 配置，确认未覆盖的训练终止项会保留。
- 未运行新仿真，未修改 SONIC 仓库与评测逻辑。
