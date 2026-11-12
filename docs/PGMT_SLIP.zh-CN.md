# PGMT P3：T3 Slip

日期：2026-10-14。按已记录公式实现持续接触滑移惩罚，权重 −1.0。T3 与 T2 使用同一个全对象接触状态 c，T1 继续使用地形向上力承重状态。

## 计算口径

$$
c_{t,f}=\mathbf1[\|F^{\mathrm{normal},w}_{t,f}\|_2>10\ \mathrm N],\qquad
f_{T3}=\sum_{f\in\{L,R\}}c_{t,f}(v_{x,t,f}^2+v_{y,t,f}^2),
$$

$$
\rho_{T3}=-1.0\times f_{T3},\qquad r_{T3}=\Delta t\times\rho_{T3}.
$$

- F 为全部接触对象对脚作用的法向力合力，使用共享 `contact/contact_valid`；阈值严格大于 10 N。侧向接触、非地形对象接触同样参与。
- v 为实际左右 `ankle_roll_link` **link 原点在世界坐标系的线速度**，来自 `body_link_lin_vel_w`。IsaacLab 的实现已包含从质心到 link 原点的角速度修正。
- 只取世界 XY 分量的平方和，再对左右脚求和。没有速度阈值、上限裁剪、两脚平均、参考速度相减或地形切平面投影。
- 每控制步持续计算，首次有效测量即可评价；参考标签无效、参考要求摆动或没有 touchdown 事件都不关闭实际接触脚的滑移惩罚。无实际接触的脚贡献 0。
- 纯竖直运动不产生本项代价；绕足底接触点的纯转动不能由该 link 原点 XY 速度完整描述。当前采用的就是原候选的速度定义。
- 数值核返回**非负、未加权代价率**，负权重 −1.0 与 dt 由 RewardManager 各乘一次。当前 dt=0.02 s：一只接触脚水平速度为 0.5 m/s，贡献为 `−1×0.5²×0.02=−0.005`；两脚均如此为 −0.01。

实际力缺测或速度出现 NaN/Inf/溢出时，该脚诊断为无效，数值核先屏蔽再做平方运算，不通过 `NaN×0` 避开异常；未知数据的代价隔离为 0，`valid=False`，不当作“有效且没有滑移”。两脚求和溢出也明确标为无效。该隔离不替代环境的非有限状态处理，本次未修改终止条件。

## 实现和时序

- [数值核](../pmt_tasks/pgmt_slip.py)：`evaluate_slip(contact, contact_valid, velocity_w)`。
- [运行期](../pmt_tasks/pgmt_runtime.py)：从已冻结接触快照的 `foot_body_lin_vel` 计算，并缓存 `rewards()["terrain_slip"]`，每控制步仅结算一次。
- `state.slip_state(for_reward=True)` 返回本步旧回合结算快照；未结算当前步时拒绝读取旧数据。`state.slip_state()` 返回当前可读副本。
- 局部 reset 清空对应行的门控/代价，速度及平方速度记为 NaN；保留其他环境与本步已结算的终止快照。重复读取不采样传感器、不推进事件、不重复积分。支持 `env_ids`，返回独立副本。
- 不新增地形射线查询，不读取增强地图、未来参考帧或离线支撑标签进行门控。

调试字段：`velocity_w=[N,2,3]`；`contact/contact_valid/velocity_valid/valid/active/speed_xy_sq/per_foot=[N,2]`；`rate=[N]`。脚序左、右；附带 `sample_step/episode_step/episode_id/reference_motion_ids/reference_frame_ids`。

## 启用边界

[Stage2 配置](../pmt_tasks/env_cfgs/pgmt/stage2.py) 默认 `pgmt_slip=True`，计算并缓存 T3；Stage1 不启用。

正式 `pgmt_stage2` 的 RewardManager 保持原 **21 项/三组**。T1/T2/T3 的正式 PPO 奖励接入按 P3 计划与四价值头/迁移一起实施。

[PGMTSlipRewardsCfg](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py) 用于独立 RewardManager 验收：原 21 项加 T1/T2/T3，共 **24 项、四组**，不创建或训练四头策略。

```bash
bash scripts/validate_pgmt_contact_state.sh --slip \
  --log_dir logs/pgmt_p3_slip/$(date +%Y%m%d_%H%M%S)
```

入口为有限步 headless 仿真，完成后自动退出，不更新模型或动作文件。

## 验证

CPU/CUDA：**125 passed**，包含 T3 和 T1/T2、共享状态/标签、原 21 项奖励、配置、课程、地图与参考回归。T3 覆盖单双脚接触、严格阈值、侧向/向下接触、XY 平方求和、纯竖直运动、世界 yaw 旋转不变性、无速度死区、正代价/负权重、无效数据和溢出、参考无效仍惩罚、首次测量/持续接触、重复读取及终止/局部 reset。纯数值核 8,192 环境的 CPU/CUDA 对照通过；不等同大规模 Isaac 训练验收。

Isaac：**passed**，见 [机器报告](../logs/pgmt_p3_slip/isaac_01/training_status.json)。五类地形各 1 环境，L0，共 72 个控制步，完成后进程自动退出。

- 独立 NumPy 根据真实接触力和脚部速度计算代价，最大误差 **9.43×10⁻⁸**；496 次有效接触单脚样本均有非零 XY 速度，224 次非接触样本贡献为 0。
- 用原生质心速度、角速度及局部质心偏移独立重建 `v_link = v_com − ω × R r_com`。排除本步已经 reset 的行后，共核对 718 次单脚速度，最大误差 **1.70×10⁻⁷ m/s**，确认未误用质心线速度。
- 非地形高台产生的 54 次全对象接触样本仍参与 T3；T1 保持地形过滤语义。T2 的双摆动/单脚有效/全部未知等内存标签夹具期间，T3 继续只按实际接触与速度计算。
- 原生 RewardManager 共 24 项；T3 的权重 −1.0、dt 和负号只生效一次，累计积分贡献约 **−0.444974**。四组总和、终止快照、局部 reset、首次测量及重复读取通过。
- T1/T2 同次联合复验通过。T1 另有五类十级的 900 条几何射线对照；T3 的动态验证范围为五类 L0，不宣称完整十级训练或收敛验收。
