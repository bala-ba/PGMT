# PGMT P3：T6 Contact force

日期：2026-10-14。实现左右脚接触力平方和惩罚，权重 −1e−6。

## 计算口径

$$
f_{T6}=\sum_{f\in\{L,R\}}\|F^{\mathrm{normal},w}_{t,f}\|_2^2
=\sum_{f\in\{L,R\}}(F_{x,t,f}^2+F_{y,t,f}^2+F_{z,t,f}^2),
$$

$$
\rho_{T6}=-10^{-6}f_{T6},\qquad r_{T6}=\Delta t\,\rho_{T6}.
$$

- F 为实际左右 `ankle_roll_link` 的**全对象法向接触力合力，世界坐标系**，来自共享快照的 `net_forces_w`。包括地形、其他对象及侧向接触，不额外加入摩擦力通道。
- XYZ 三分量均参与平方；先对每只脚求平方和，再对左右脚求和。左右脚方向相反时不会相互抵消。每只脚内部仍使用原生传感器的合力，不是逐接触点的力平方之和。
- **无阈值、无上限、无接触状态门控**：不减去 350 N，也不乘 `c=1[||F||>10 N]`。任何有限有效的非零力都产生代价，包括 10 N 以下的力；零力贡献为 0。
- 每控制步持续计算，首次有效测量即可评价。不依赖参考标签、地形承重、touchdown、历史年龄、扫描掩膜或足部速度。
- 不改变 A10 `auxiliary_anchor_contact_force`：A10 对 torso 计算，本项只对左右脚计算，各用自身权重。
- 数值核返回非负、未加权代价率，RewardManager 乘 −1e−6 和 dt 各一次。当前 dt=0.02 s：单脚 450 N 为 **−0.00405**，两脚均为 450 N 时 **−0.0081**；单脚 5 N 为 **−5×10⁻⁷**。

实际力缺测、NaN/Inf 或平方运算溢出时，该脚诊断为 `valid=False`，代价隔离为 0；未知力不解释为有效零力。先屏蔽无效输入再平方，避免 `NaN×0`。两脚各自平方有限、但总和溢出时，两脚最终 `valid` 均为 False、代价率为 0，保留各自有限的 `force_sq` 供定位；正常有限值不裁剪。本项未改变环境终止条件。

力按 50 Hz 控制步末读数计算，没有改为 200 Hz 物理子步峰值、平均值或力冲量；该项不保证捕捉所有短于控制周期的冲击。

## 实现与缓存

- [数值核](../pmt_tasks/pgmt_contact_force.py)：`evaluate_contact_force(forces_w, force_valid)`。`force_valid` 使用共享 `contact_valid` 的测量有效性，不是 `contact` 的布尔接触状态。
- [运行期](../pmt_tasks/pgmt_runtime.py)：共享接触快照冻结后计算 `rewards()["terrain_contact_force"]`，每控制步只结算一次。
- `state.contact_force_state(for_reward=True)` 返回本步旧回合的力、平方代价与时间信息；尚未结算当前步时拒绝读取旧数据。默认接口返回当前副本，支持 `env_ids`，读者修改副本不会改变缓存。
- 局部 reset 清空对应行的有效性和代价，力与平方力记为 NaN；其他环境、本步已结算的终止快照保持不变。重复读取不采样传感器、不推进事件、不重复积分。不新增地形射线或感知处理。

调试字段：`forces_w=[N,2,3]`；`force_valid/force_sq/valid/per_foot=[N,2]`；`rate=[N]`。脚序左、右，附带 `sample_step/episode_step/episode_id/reference_motion_ids/reference_frame_ids`。

## 启用边界与复验

[Stage2](../pmt_tasks/env_cfgs/pgmt/stage2.py) 默认 `pgmt_contact_force=True`，计算并缓存 T6；Stage1 不启用。

正式 `pgmt_stage2` 的 RewardManager 仍为原 **21 项/三组**。六项 terrain-contact 的正式 PPO 接入按 P3 计划与四价值头/迁移一起实施。

[PGMTContactForceRewardsCfg](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py) 用于独立验收：原 21 项加全部 T1–T6，共 **27 项、四组**，不创建或训练四头策略。

```bash
bash scripts/validate_pgmt_contact_state.sh --contact_force \
  --log_dir logs/pgmt_p3_contact_force/$(date +%Y%m%d_%H%M%S)
```

有限步 headless 仿真完成后自动退出，不更新模型或动作文件。

## 验证记录

CPU/CUDA：**223 passed**，包括 T6、T1–T5、共享接触状态/标签、原 21 项奖励、配置、课程、地图和参考回归。T6 覆盖 XYZ 平方和、左右脚独立累加、旋转/符号不变性、0/弱力/10 N/350 N/450 N、首次测量与持续计罚、参考无效、非地形力、NaN/Inf/单脚及总和溢出、终止/局部 reset 和重复读取。8,192 环境的数值核 CPU/CUDA 对照通过，不等同大规模训练验收。

Isaac：**passed**，见 [机器报告](../logs/pgmt_p3_contact_force/isaac_01/training_status.json)。五类地形各 1 环境、L0，共 **102 个控制步**，含非地形高台、参考标签夹具和真实侧撞场景。进程 334012 自动退出并已确认不存在。

- 原生 RewardManager 共 **27 项、四组**，全部六项 terrain-contact 联验通过；T6 累计贡献约 **−0.297415**，权重 −1e−6 与 dt 各乘一次，四组总和等于标量奖励。
- [独立验收器](../scripts/pgmt_contact_force_checks.py) 按足部名称重新定位原生全对象传感器列，在非 reset 行核对源数据，再用 NumPy float64 独立计算 XYZ 平方和。代价率最大绝对误差 **0.03433 N²**，最大相对误差 **1.15×10⁻⁷**，符合 float32 对照精度。
- 552 个非零力单脚样本产生代价，468 个零力样本贡献为 0；其中 **24 个非零力样本不超过 10 N**，仍产生代价，验证没有接触阈值门控。9 个样本超过 350 N，最大单脚力约 **820.644 N**，使用完整力平方而非超量。
- 58 个非地形力样本、6 个参考标签无效但实际有力的样本参与 T6。终止步冻结、局部 reset、首次测量及重复读取通过；没有新增地形查询。
- T1–T5 同次联合复验通过。T1 另含五类十级的 100 个窗口/900 条几何射线对照；T6 动态范围是五类 L0 和上述物理场景，不宣称完成全十级训练或收敛验收。
