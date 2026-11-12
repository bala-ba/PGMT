# PGMT P3：T4 Stumble

日期：2026-10-14。实现左右脚水平接触力异常惩罚，权重 −20。

## 计算口径

$$
b_{t,f}=\mathbf1[\sqrt{F_{x,t,f}^2+F_{y,t,f}^2}>5|F_{z,t,f}|+10\ \mathrm N],
\qquad f_{T4}=\sum_{f\in\{L,R\}}b_{t,f},
$$

$$
\rho_{T4}=-20 f_{T4},\qquad r_{T4}=\Delta t\,\rho_{T4}.
$$

- F 是实际左右 `ankle_roll_link` 的**全对象法向接触力合力，世界坐标系**，来自共享接触快照的 `net_forces_w`。包括非地形对象接触，不限于地形向上作用力；不额外加入摩擦力。
- 使用世界 XY 力范数和世界 Z 力绝对值；判据是严格 `>`，等于阈值不计罚。正负 Fz 的绝对值相同则阈值相同。
- 逐脚产生 0/1，左右求和得到 0/1/2；既不对两脚取平均，也不先对两脚合力做判断。
- **每控制步持续评价**，首次有效读数即可计算，条件持续成立则持续计罚。没有 touchdown、事件上升沿、历史年龄、参考支撑标签、地形掩膜或速度门槛。默认参数下满足本判据已经意味着全对象力范数大于 10 N，无需额外乘接触状态 c。
- `STUMBLE_CONFIG={"vertical_ratio":5.0,"bias_n":10.0}` 集中保存比例和偏置，要求两者均为有限非负数。它们是本地实现采用的工程初值，不能称为论文公开常数。
- 数值核返回非负、未加权代价率。RewardManager 只乘一次 −20 和 dt；当前 dt=0.02 s，单脚触发每步 **−0.4**，双脚触发每步 **−0.8**。

异常测量（NaN/Inf、共享力测量无效、范数或阈值运算溢出）单脚标为 `valid=False`，代价隔离为 0，力范数/阈值诊断为 NaN。不把未知数据当成有效且没有绊碰；本项未改变环境终止条件。正常有限值不做裁剪。

这是控制步末 50 Hz 的读数，不宣称捕捉到每个 200 Hz 物理子步内的短暂侧撞。脚上多个对象的力先由原生传感器求合力，方向相反的力可能抵消，当前没有逐接触点判定。

## 代码与时序

- [数值核与参数](../pmt_tasks/pgmt_stumble.py)：`evaluate_stumble(forces_w, force_valid, config)`。`force_valid` 是测量有效性，不是接触或承重状态。
- [运行期](../pmt_tasks/pgmt_runtime.py)：共享状态冻结后计算 `rewards()["terrain_stumble"]`，每控制步只结算一次，不增加地形查询或独立传感器采样。
- `state.stumble_state(for_reward=True)` 返回本步旧回合结算快照；尚未结算当前步时拒绝读取旧数据。`state.stumble_state()` 返回当前副本，支持 `env_ids`。
- 局部 reset 清空对应行的有效性、触发标记和代价；力、范数、阈值重置为 NaN。保留其他环境以及本步终止奖励快照。读取返回独立副本，不更新接触历史、不重复积分。

调试字段：`forces_w=[N,2,3]`；`force_valid/horizontal_force/vertical_force_abs/threshold/valid/stumble/per_foot=[N,2]`；`rate=[N]`。脚序左、右，附带 `sample_step/episode_step/episode_id/reference_motion_ids/reference_frame_ids`。

## 启用边界与复验

[Stage2 配置](../pmt_tasks/env_cfgs/pgmt/stage2.py) 默认计算、缓存 T4；`pgmt_stumble=None` 可关闭，Stage1 不启用。

正式 `pgmt_stage2` 的 RewardManager 保持原 **21 项/三组**。T1–T4 的正式 PPO 接入按 P3 计划与四价值头/迁移一起实施。

[PGMTStumbleRewardsCfg](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py) 用于独立验收：原 21 项加 T1–T4，共 **25 项、四组**，不创建或训练四头策略。

```bash
bash scripts/validate_pgmt_contact_state.sh --stumble \
  --log_dir logs/pgmt_p3_stumble/$(date +%Y%m%d_%H%M%S)
```

有限步 headless 仿真完成后自动退出，不更新模型或动作文件。

## 验证记录

CPU/CUDA：**164 passed**，含 T4 和 T1–T3、共享接触状态/标签、原 21 项奖励、配置、课程、地图和参考回归。T4 覆盖严格阈值、向下/纯水平/纯竖直力、双脚求和、yaw 旋转不变性、首次测量与连续计罚、未知参考/非地形接触、异常值与溢出、终止/局部 reset、重复读取。纯数值核 8,192 环境 CPU/CUDA 对照通过，不等同大规模 Isaac 训练验收。

Isaac：**passed**，见 [机器报告](../logs/pgmt_p3_stumble/isaac_02/training_status.json)。五类地形各 1 环境、L0，共 **102 个控制步**；进程 290332 自动退出并已确认不存在。

- 原生 RewardManager 共 25 项；独立 NumPy 用真实冻结接触力计算，T4 代价率最大误差 **0**。1018 个单脚样本不触发，2 个单脚样本触发；累计积分贡献约 **−0.8**。负权重、dt 各乘一次，四组总和等于标量奖励。
- 首轮 [isaac_01](../logs/pgmt_p3_stumble/isaac_01/training_status.json) 的 72 步没有自然侧撞，触发数为 0，因缺少正例而明确失败。随后增加仅存在于验收脚本中的真实侧向障碍物，不修改训练地形或覆盖传感器力。
- 机器人保持完整关节姿态，足部碰撞覆盖距障碍物侧面 15 mm，分别以 0.5/1/2 m/s 的初始世界 X 速度接近，各运行 10 步。0.5 m/s 场景在控制步采样中未触发；1 m/s 时右脚出现 `F=(-47.8159,0,0) N`，2 m/s 时左脚出现 `F=(-98.9283,0,0) N`，均超过 10 N 阈值，各产生 −0.4。两次均没有地形承重或 touchdown，验证了全对象力语义。
- 持续触发的逐步计罚由数值核和运行期测试覆盖；本次真实侧撞是两个单步触发，未声称测得持续绊碰或全部物理子步事件。
- 终止步冻结、局部 reset 清空当前副本、重复读取不推进状态、参考标签切换与无效标签夹具期间独立计算均通过。T1–T3 联验通过；T1 另有五类十级的 100 个窗口/900 条几何射线对照。T4 动态范围是五类 L0 和独立侧撞场景，不等同完整十级训练或收敛验收。
