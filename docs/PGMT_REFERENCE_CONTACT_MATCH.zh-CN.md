# PGMT P3：T2 Reference contact match

日期：2026-10-14。用户确认采用原候选的**全对象接触力范数**判据。T2 为持续奖励，复用共享接触快照和同一参考帧的离线 TCRS 标签；不依赖 touchdown 事件，也不增加地形查询。

## 公式与掩码

每只脚的实际状态 `c = (||F_net_normal_world||₂ > 10 N)`，阈值严格大于。复用 `contact/contact_valid`；F 为所有接触对象的法向力合力，包含非地形碰撞，未加入摩擦力。侧向或向下接触力达到阈值也可令 c=1。T1 仍采用地形过滤力世界 Z 分量的承重判据，两个奖励不共用同一个布尔状态。

参考状态 `cʳ` 和有效性 `mʳ` 来自当前动作/帧的 `ContactLabelStore`，脚序为左、右。令 `v` 为实际力测量有效性：

$$
f_{T2}=\frac{\sum_{f\in\{L,R\}}m^r_f v_f\,\mathbf1[c_f=c^r_f]}
                 {\max(\sum_{f\in\{L,R\}}m^r_f,1)},\qquad
r_{T2}=\Delta t\times1.5\times f_{T2}.
$$

- 有效参考支撑与摆动都参与匹配，每个控制步重新结算；首次有效测量即可评价，持续匹配持续得分。
- 两脚参考均有效：全部匹配率为 1，仅一脚匹配为 0.5，全部不匹配为 0。当前 dt=0.02 s，对应每步贡献 0.03、0.015、0。
- 仅一脚参考有效：只评价该脚，匹配率为 1 或 0；两脚参考均无效时为 0。
- 实际力 NaN/Inf 对应 `actual_valid=False`，该脚不获得匹配分，也不缩小有效参考数量这一分母。例如两脚参考有效、仅一脚实际可读且匹配，率为 0.5；不能因传感器缺测得到满分。
- 参考 unknown 不当作 swing。源文件在载入时验证为二值；数值核对诊断调用的 NaN/Inf/非二值标签也屏蔽，不产生 NaN 奖励。
- 有效性来自参考标签和实际测量，不使用策略地图 dropout、扫描覆盖、T1 九点掩码或 touchdown 历史门槛。

## 实现与时序

- [数值核](../pmt_tasks/pgmt_contact_match.py)：`evaluate_reference_contact_match()` 返回未加权率，不乘 weight/dt。
- [运行期](../pmt_tasks/pgmt_runtime.py)：在 `_update_contact_state()` 冻结本控制步接触/参考后计算 T2；`rewards()["terrain_reference_contact_match"]` 每步缓存一次。
- `state.reference_contact_match_state(for_reward=True)` 返回旧回合结算快照；未结算当前步时拒绝读取，避免误用上一控制步数据。
- 默认 `state.reference_contact_match_state()` 返回当前可读副本；局部 reset 清空对应环境的率和有效性，保留其他环境及已结算终止步快照。支持 `env_ids`，所有返回字段为独立副本。
- 标签取自结算时实际 `motion_ids/frame_ids`，不读取 command 推进后的下一帧，不用未来帧或 reset 后第 0 帧替代终止帧。

调试字段：`actual_contact/actual_valid/reference_contact/reference_valid/compared_valid/matched=[N,2]`；`reference_count/rate=[N]`；同时带 `sample_step/episode_step/episode_id/reference_motion_ids/reference_frame_ids`。无效状态须结合对应掩码读取。

## 启用边界

[Stage2 配置](../pmt_tasks/env_cfgs/pgmt/stage2.py) 的 `pgmt_reference_contact_match=True` 默认计算缓存；Stage1 不启用该项。

原 `pgmt_stage2` 基线的 RewardManager 保留 **21 项/三组**，T1/T2 不加入该基线的 PPO 总奖励；正式 `pgmt_stage2_terrain` 已在 P3.4 同步启用 terrain 组、四头与迁移，T1/T2 均归入 terrain。

[独立奖励配置](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py) 的 `PGMTContactMatchRewardsCfg` 包含原 21 项及 T1/T2，共 **23 项、四组**，用于真实 RewardManager 验收，不创建或训练四头策略。

```bash
bash scripts/validate_pgmt_contact_state.sh --contact_match \
  --log_dir logs/pgmt_p3_contact_match/$(date +%Y%m%d_%H%M%S)
```

入口为有限步 headless 检查，结束后自动退出，不更新 checkpoint 或动作源文件。

## 验证

CPU/CUDA：**107 passed**，包含 T2 与 T1、接触状态/标签、原 21 项奖励、配置、课程、地图和参考回归。T2 覆盖所有二值状态/参考掩码组合、严格 10 N 阈值、侧向/向下/非地形接触、未知测量、无效参考、动作切换/片段边界、持续匹配、终止快照、重复读取及局部 reset；8,192 环境的纯数值 CPU/CUDA 结果逐值一致。该规模是数值核测试，不是 8,192 环境 Isaac 训练验收。

Isaac：**passed**，见 [机器报告](../logs/pgmt_p3_contact_match/isaac_01/training_status.json)。五类地形各 1 环境，L0，共 72 个控制步；独立 NumPy 从冻结的真实全对象力计算匹配，与 T2 数值核最大误差 **0**。700 次有效单脚评价、494 次匹配，累计 T2 加权积分贡献约 7.455。

- 每次 `env.step()` 前独立捕获参考标签及 motion/frame，与结算快照逐项核对，覆盖 command 推进和终止 reset。
- 原生 RewardManager 的实际项数为 23；T1/T2 权重和 dt 均只乘一次，四组和等于原生标量奖励，重复读取不重算。
- 额外 5 步使用明确标注的内存标签夹具，分别检查双支撑、双摆动、仅左有效、仅右有效和全部未知；真实物理力保持不变，结束恢复标签内存，未改动作 NPZ。
- 非地形高台对照中，全对象力最高约 820.64 N、地形过滤力为 0；共观察到 54 个接触/承重状态不同的单脚样本，T2 使用全对象接触，T1 无伪地形落脚奖励。
- 同次重跑 T1 五类十级的 100 个几何窗口/900 条独立射线，并核对 13 次实际落脚，仍通过。T2 动态仿真范围为五类 L0，不宣称覆盖全部十级动态训练。

验收进程自动退出；未启动 GUI/PPO，不代表策略收敛或大规模训练性能验收。
