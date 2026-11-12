# PGMT 身体碰撞到奖励的原生验收

日期：2026-10-14。**真实碰撞归属、接触力传递、奖励计算、四组归集与 reset 缓存通过；发现控制步末采样会漏掉短冲击峰值。** 本次只新增验收脚本和文档，未修改生产奖励、权重、物理参数、资产或动作。膝/手必要接触的动作语义统计仍待完成，P3.5 不能因此整体标记完成。

## 1. 运行与证据

入口：[validate_pgmt_body_contacts.py](../scripts/validate_pgmt_body_contacts.py)、[启动脚本](../scripts/validate_pgmt_body_contacts.sh)。实际使用正式 `pgmt_stage2_terrain` 的 27 项奖励和四组配置、原生 USD、RTX 5090 `cuda:0`、5 个环境；44 控制步、176 物理子步，共 220 个环境 transition，无 PPO 更新。成功进程 PID 633725 已退出。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash scripts/validate_pgmt_body_contacts.sh \
  --log_dir logs/pgmt_body_contacts/isaac_new
```

必须使用新的输出目录，脚本拒绝覆盖旧结果。默认 headless，结束后自动关闭 Isaac。

- [原生运行结果](../logs/pgmt_body_contacts/isaac_06/training_status.json)：`passed` 表示下述计算断言通过。
- [独立分析结果](../logs/pgmt_body_contacts/isaac_06/analysis_report.json)：`passed_with_sampling_limitation`，明确记录采样问题。
- [逐步奖励](../logs/pgmt_body_contacts/isaac_06/reward_records.json)、[真实 GPU 接触点](../logs/pgmt_body_contacts/isaac_06/physical_contacts.json)、[每个物理子步的力](../logs/pgmt_body_contacts/isaac_06/physics_samples.json)、[完整张量](../logs/pgmt_body_contacts/isaac_06/samples.npz)。

张量分别为力 `[44,5,30,3]`、原始奖励及实际贡献 `[44,5,27]`、四组回报 `[44,5,4]`；原始传感器在 RewardManager 计算前、自动 reset 前读取。NumPy 使用 float64 独立重算。

测试控制初始位姿、速度，加入一个高于地面的真实小障碍物及只读 torso–obstacle 过滤传感器；关闭质量/姿态扰动和随机推力以隔离探针。保留正式终止条件，额外添加仅供验收的“实测 torso 接触力 >10 N 时终止”开关，用于保证碰撞当步发生 reset。没有写入或伪造传感器力，没有修改奖励函数。

## 2. 验证的计算链路

1. **真实物理接触点 → 碰撞体**：过滤对象为障碍物实际碰撞 prim `/World/PGMTBodyAuditObstacle/geometry/mesh`。将 GPU PhysX 报告的接触点变换至 torso 局部系，与实际 USD 四个 capsule 的表面距离比较。10 个法向力大于 1 N 的点均满足目标 capsule 距离 <3 cm、其他 capsule 距离 >3 cm，可唯一识别本次探针接触部位。路径归属由接触点几何核验得到，并非 CPU 接触事件返回的 collider ID。
2. **接触点 → 合力**：逐点 `Σ f_normal × n` 与障碍物过滤力矩阵一致；隔离碰撞中该矩阵又与生产 `contact_forces.data.net_forces_w` 的 torso 合力一致。保留法向力定义，不混入摩擦力。
3. **传感器 → 正确身体索引**：传感器数组和 articulation 数组顺序不同，运行时通过名称映射。检查完整 30 身体映射、`torso_link` anchor 和排除双脚后的 28 身体集合。
4. **原始 rate → 实际 reward**：检查两项独立公式，并验证正式 27 项的权重与 dt、四组和及最终标量。
5. **终止/reset → 缓存**：在真实头部接触步触发终止；旧回合奖励保持，新回合接触有效性清零，重复读取不推进历史；下一物理控制步移动至空中后两项惩罚归零。

公式保持现状，`Δt=0.02 s`：

\[
r_{undesired}=-0.1\,\Delta t\sum_{b\notin\{left\_foot,right\_foot\}}
\mathbf{1}[\|F_b\|_2>10\ \mathrm{N}]
\]

\[
r_{impact}=-10^{-5}\,\Delta t\,\|F_{torso}\|_2^2.
\]

头部 `mesh_3/capsule` 和胸部 `mesh_0/capsule` 都归属 `torso_link`。本次隔离撞击中只有 torso 被计为 undesired，因此此项每个命中步为 **−0.002**，不会按多个接触点或额外 head 节点重复计数。同一撞击同时触发 undesired 计数和 impact 力平方，是两项不同的现有惩罚。

## 3. 数值结果

下表为各 8 控制步探针中的最大值；“物理峰值”来自 200 Hz 逐子步读取，“奖励样本”来自正式 50 Hz 奖励时刻，不能互换。

| 探针 | 目标 capsule | 物理峰值 N | 奖励样本最大合力 N | 该奖励步 impact 贡献 |
| --- | --- | ---: | ---: | ---: |
| 头部，初速 1 m/s | torso mesh_3 | 659.982 | 659.982 | −0.0871153 |
| 头部，初速 2 m/s | torso mesh_3 | 1798.678 | 58.553 | −0.000685692 |
| 胸部，初速 1 m/s | torso mesh_0 | 2249.371 | 2249.371 | −1.0119337 |
| 胸部，初速 2 m/s | torso mesh_0 | 5115.243 | 252.508 | −0.0127520 |

空中无接触负例通过；平地双脚承重中有 3 个仅脚受力的控制步，两项身体惩罚均为零。四组正例及终止碰撞均通过。

impact 原始 rate 相对 NumPy 最大误差 **4.634×10⁻⁸**；两项最终加权贡献最大绝对误差 **9.710×10⁻⁸**；27 项独立重算与总奖励最大误差 **8.315×10⁻⁸**；四组和与标量奖励最大误差 **4.470×10⁻⁸**。

## 4. 发现与处理

### 短冲击的控制步末采样问题：保留现状并登记

物理 `dt=0.005 s`、`decimation=4`。生产奖励读取最后一个物理子步的 `net_forces_w`，不是四个子步的峰值或积分。因此，短冲击可能在前几个子步发生并消退，奖励采到较小的尾部力；也可能恰好采到峰值，再按完整 0.02 s 乘 dt。**当前公式计算正确，不等于完整捕获每次冲击强度。** 本次 2 m/s 的两个探针直接出现该问题，也不能要求当前惩罚随入射初速度单调增大。

若后续决定将力平方惩罚按真实物理时间积分，可在四个子步累计 `||F||²` 后取均值，再让 RewardManager 乘一次 0.02 s，等价于 `Σ ||F_k||²×0.005 s`。使用子步最大值则是另一种惩罚定义。两者都涉及采样契约变更；本次未擅自选择或修改。`undesired_contacts` 的短接触计数如何聚合也需同步明确。

### reset 后原生传感器仍可读到旧物理结果：PGMT 隔离通过

本次终止步 reset 后，直接读取原生传感器仍得到 torso 约 **58.552 N**。原生 SensorBase reset 将缓存标为待更新；再次读取会查询尚未经过新物理步的 PhysX 结果，不能要求瞬间归零。

PGMT 的有效性和奖励缓存正确隔离这一时间差：新回合 `contact_valid=False`，当前已结算奖励仍属于旧回合；下一次真实物理步重新取样，空中力及两项惩罚均为零。未发生跨回合重复计罚。`training_status.json` 的 `post_reset_stale_force_absent` 对应这个下一物理步检查，不代表原生缓冲区在 reset 瞬间必为零。

### 验收工具适配：已解决

- `isaac_01`～`isaac_03`：GPU 力存在，但本机 CPU 接触事件/同步报告均为空，开启通知也没有获得 collider 证据；不把这些运行算作通过。最终改为 GPU 接触点、过滤力与实际胶囊几何交叉验证，不依赖 CPU 报告。
- `isaac_04`：审计过滤器最初指向障碍物 Xform，未选中静态碰撞体；修正为实际 `geometry/mesh`。生产地形过滤器原本已使用实际 mesh，本次未改动。
- `isaac_05`：四组碰撞已通过，但“reset 后原生力立即为零”的错误测试假设失败；改为验证 PGMT 有效性、旧奖励快照及下一物理步，无需改生产逻辑。
- `isaac_06`：完整 44 步通过，同时记录采样限制。生产 Python 文件与 `docs/USER_READ.md` 前后哈希完全一致，见 [之前](../logs/pgmt_body_contacts/source_hashes_before.json) / [之后](../logs/pgmt_body_contacts/source_hashes_after.json)。

## 5. 代码位置与剩余范围

- [PGMTState 初始化及 sensor 顺序映射](../pmt_tasks/pgmt_runtime.py)：`__init__()` 中 anchor、`contact_order`；`rewards()` 读取真实传感器并缓存。
- [两项原始 rate](../pmt_tasks/pgmt_math.py)：`reward_terms()` 最后两项；`body_groups()` 定义 undesired 身体集合。
- [RewardManager 适配](../pmt_tasks/mdp/pgmt_rewards.py)：`pgmt_reward()` 返回无权重、无 dt 的 rate。
- [四组归集](../pmt_tasks/pgmt_runtime.py)：`reward_groups_from_manager()`；[训练 wrapper](../pmt_tasks/isaaclab_rl/rsl_rl/vecenv_wrapper.py) 将其接入 PPO rollout。

本次关闭了“头/胸是否真实进入正确奖励项”的验证缺口；未进行爬行膝/手合理接触统计、策略性能评估、PPO 重训或扩容。短冲击采样风险已用实际物理样本确认，后续应明确所需聚合语义。
