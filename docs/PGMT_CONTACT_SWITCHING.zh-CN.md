# PGMT P3：T5 Contact switching

日期：2026-10-14。复用共享接触状态的快速切换事件，权重 −30。

## 公式与时间口径

$$
c_{t,f}=\mathbf1[\|F^{\mathrm{normal},w}_{t,f}\|_2>10\ \mathrm N],
\quad S_{t,f}=b^c_{t,f}\mathbf1[c_{t,f}\ne c_{t-1,f}],
$$

$$
K_{t,f}=S_{t,f}\mathbf1[A_{t-1,f}<0.1\ \mathrm s]\mathbf1[n_t>5],
\quad f_{T5}=\sum_{f\in\{L,R\}}K_{t,f},
\quad r_{T5}=\Delta t(-30f_{T5}).
$$

- F 为左右 `ankle_roll_link` 的全对象世界法向接触力合力，和 T2/T3 的实际接触 c 一致。包含侧向、非地形对象的接触；不使用地形向上力承重 s 或参考支撑标签门控。
- `bᶜ = contact_history_valid` 表示前后测量有效、同一回合且已有历史。进入接触和退出接触两个方向都检查。
- `A` 是共享状态中的接触状态年龄：首次有效测量及切换当步记 0，状态保持时每步累加 dt。T5 读取**切换前的年龄**，不读取刚归零的新年龄，也不使用另一个口径的 `contact_time/air_time`。
- 阈值严格小于 0.1 s；恰好 0.1 s 不计罚。共享状态使用整数步龄进行比较，避免 float32 在边界上误判；T5 不再重复一次浮点时间比较。
- `n` 从回合第 1 步开始计数；前 5 步不计快速切换惩罚，第 6 步起生效。初始保护不重置正常年龄累计，也不影响其他奖励。
- **只在符合条件的切换当步计罚**。保持接触、保持不接触，以及切换前状态年龄达到 0.1 s 的正常切换，都不会发放本项惩罚。短接触或短间隔后再次切换可各自触发，左右脚独立求和。
- 接触阈值、快速切换阈值和初始保护统一读取 `CONTACT_STATE_CONFIG` 的 `contact_threshold_n=10.0`、`fast_switch_seconds=0.1`、`switch_grace_steps=5`。T5 没有新建计时器或另一套阈值。未公开的这些数值属于工程设定。
- 数值核返回非负未加权率 0/1/2，RewardManager 乘 −30 与 dt 各一次。当前 dt=0.02 s，单脚触发一次 **−0.6**，双脚同时触发 **−1.2**。

例如同一脚从第 1 至第 6 步保持接触，第 7 步离开：旧年龄为 0.10 s，不罚；第 8 步重新接触：旧非接触状态年龄为 0，满足快速切换条件，扣 0.6。这个例子采用已有的“切换当步年龄归零”定义，不把年龄替换为连续观测样本数乘 dt。

NaN/Inf 或力范数溢出会中断相应脚的接触历史。缺测当步和恢复后的首次有效测量均不形成切换事件；恢复后连续有效的实际切换可正常计罚。无历史时 `valid=False`、代价为 0，不能当成已确认没有切换。训练期不额外读取参考标签有效性、增强地图或地形查询。

## 实现与缓存

- [共享事件源](../pmt_tasks/pgmt_contact_state.py)：`ContactEventState._advance_stream()` 生成 `fast_contact_switch`。本次复用已有年龄、保护和事件规则。
- [数值核](../pmt_tasks/pgmt_contact_switching.py)：`evaluate_contact_switching(fast_contact_switch, contact_history_valid)`。
- [运行期](../pmt_tasks/pgmt_runtime.py)：共享接触快照冻结后计算并缓存 `rewards()["terrain_contact_switching"]`，每控制步结算一次。
- `state.contact_switching_state(for_reward=True)` 返回本步旧回合快照；尚未结算当前步时拒绝读取旧奖励。默认 `for_reward=False` 返回当前副本，支持 `env_ids`，所有输出为独立副本。
- 局部 reset 清理对应行的事件、有效性和代价，保持其他环境和本步终止奖励。重复读取不推进年龄、不补发事件、不重复积分。

调试输出包括 `fast_contact_switch/valid/per_foot=[N,2]`、`rate=[N]`，以及共享快照的 `contact/previous_contact/contact_valid/contact_history_valid/contact_age/previous_contact_age/contact_started/contact_ended/contact_switched` 和 `sample_step/episode_step/episode_id/reference_motion_ids/reference_frame_ids`。脚序为左、右。

采用 50 Hz 控制步末采样；没有新增 200 Hz 物理子步事件锁存。短于控制周期且未被末帧观测到的接触切换可能遗漏。

## 启用边界与复验

[Stage2](../pmt_tasks/env_cfgs/pgmt/stage2.py) 默认 `pgmt_contact_switching=True`，计算并缓存 T5；Stage1 不启用。

正式 `pgmt_stage2` 的 RewardManager 仍为原 **21 项/三组**。T1–T5 的正式 PPO 接入按 P3 计划与四价值头/迁移一起实施。

[PGMTContactSwitchingRewardsCfg](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py) 用于独立验收：原 21 项加 T1–T5，共 **26 项、四组**，不创建或训练四头策略。

```bash
bash scripts/validate_pgmt_contact_state.sh --contact_switching \
  --log_dir logs/pgmt_p3_contact_switching/$(date +%Y%m%d_%H%M%S)
```

有限步 headless 仿真完成后自动退出，不更新模型或动作文件。

## 验证记录

CPU/CUDA：**195 passed**，包含 T5、T1–T4、共享接触状态/标签、原 21 项奖励、配置、课程、地图和参考回归。T5 覆盖两个切换方向、0.1 s 严格边界、0.02/0.025/0.03 s 步长、初始保护、事件后保持状态不重复计罚、缺测与溢出恢复、终止/局部 reset、无效参考和重复读取。8,192 环境、24 步随机接触历史的 CPU/CUDA 对照通过；该规模是纯数值验收。

Isaac：**passed**，见 [机器报告](../logs/pgmt_p3_contact_switching/isaac_01/training_status.json)。五类地形各 1 环境、L0，共 **102 个控制步**，含非地形高台、参考标签夹具和 T4 的真实侧撞场景。进程 300151 自动退出并已确认不存在。

- 原生 RewardManager 共 26 项，T5 累计贡献约 **−2.4**。真实快速切换 **4 次**：进入接触 2 次、退出接触 2 次；代价率最大误差 **0**。负权重与 dt 各生效一次，四组总和等于标量奖励。
- [独立验收器](../scripts/pgmt_contact_switching_checks.py) 从冻结的原始力重新判断接触，用 NumPy 独立保存回合、历史和整数年龄，再用 Decimal 秒值比较阈值。没有把共享 `fast_contact_switch` 本身作为预期答案；前后状态、旧/新年龄、保护期与计罚逐步对照通过。
- 13 次短状态切换被前 5 步保护排除，11 次因旧状态年龄达到阈值而不计罚；972 个保持状态的单脚样本为 0，20 个首次测量/重置基线不制造事件。第 6 控制步测得一次旧年龄 0.08 s 的进入接触，验证保护边界后可正常触发。
- 恰好 0.1 s 的切换、未知参考仍可触发及缺测恢复由数值/运行期测试覆盖；本次真实轨迹未出现这些条件下的正例，不将它们计入真实物理覆盖。
- 终止步冻结、局部 reset、重复读取和 T1–T4 联验通过。T1 另有五类十级的 100 个窗口/900 条射线对照；T5 动态范围仍为五类 L0 和上述物理场景，不等同完整十级训练或收敛验收。
