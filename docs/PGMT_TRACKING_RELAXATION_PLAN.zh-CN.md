# PGMT P3：地形跟踪容差实施计划

日期：2026-10-14。状态：**C1–C5 已完成。** 新正式任务 `pgmt_stage2_terrain` 默认启用三项容差、27 项奖励和四头；原 P2 默认关闭，Stage1 保持严格跟踪。数值专项及四头 160 环境 PPO/严格续训验收均通过，见 [P3.4 验收](PGMT_P34_FOUR_HEAD_ACCEPTANCE.zh-CN.md)。本次未修改终止阈值、延迟或课程成功判据。

依据：[论文](../PGMT_LITERATURE.pdf)第 5 页「Terrain-Aware Tracking Relaxation」、[当前奖励数值核](../pmt_tasks/pgmt_math.py)、[当前位置地形查询](../pmt_tasks/pgmt_terrain_query.py)。论文规定作用范围、逐元素死区结构和当前 tile 的难度来源；下列上限、L/9 映射、anchor 选择及初始 α 为工程设定。

## 1. 采用的公式和参数

对指定下肢目标 h 的每个 link/joint j：

$$
\widetilde e_{h,j}=\max\{0,e_{h,j}-\alpha_{h,j}\chi(\kappa_t)m_h\operatorname{clip}(L_t/9,0,1)\},
\qquad
f_h=\exp\left[-\frac{\operatorname{mean}_j(\widetilde e_{h,j}^{\,2})}{\sigma_h^2}\right].
$$

默认 `χ=1` 仅限 stairs/slope/boxes；flat/rough 为 0。现在可通过 `terrain_families` 修改适用地形列表。权重、σ、dt 约定沿用当前实现，三项仍属于 `lower` 组，不新增奖励项或价值头。

| 目标 | 基础容差上限 m | 原 σ | 原权重 | 原始误差 |
| --- | --- | --- | --- | --- |
| `lower_body_position` | 0.20 m | 0.15 m | +0.5 | 每个 link 的 anchor 相对位置三维误差范数 |
| `lower_body_orientation` | 0.70 rad | 0.40 rad | +2.0 | 每个 link 在各自 anchor yaw 对齐后的四元数夹角 |
| `lower_joint_position` | 0.50 rad | 0.40 rad | +0.5 | 每个关节的实际/参考角度差绝对值 |

容差从上表的**标量原始误差**中逐元素扣除，再平方、均值和取指数；不在 XYZ 各轴分别扣除，也不在平方或均值之后扣除。用户所述“保持 error 为正”实现为 **error ≥ 0**；小于或等于容差时归零，不额外加 epsilon。非有限原始误差仍保留非有限证据，不以容差转换成正常状态。沿用现有相对坐标与原始参考，不修改关节轨迹、统一放置变换、command、观测或全局位置修正。

首版对现有下肢 **12 个 link、12 个 joint** 设置 α=1：左右 hip_pitch/roll/yaw、knee、ankle_pitch/roll。复用 `body_groups()`，按名称解析并验证，可按名称逐元素覆写有限非负 α；α=0 恢复严格跟踪。m 是基础上限，实际单元素上限为 αm。上肢、所有速度项、auxiliary 和 T1–T6 保持现有计算。

| 当前 tile 等级 | 位置容差 / m | 姿态容差 / rad | 关节容差 / rad |
| --- | ---: | ---: | ---: |
| L0 | 0 | 0 | 0 |
| L3 | 0.0667 | 0.2333 | 0.1667 |
| L6 | 0.1333 | 0.4667 | 0.3333 |
| L9 | 0.20 | 0.70 | 0.50 |

表中取 χ=1、α=1。当前 v6 的 L0 台阶/箱体仍有实际高度，L0 零容差是承接严格基线的工程选择，不能解释为 L0 全部几何平坦；独立验收需记录其跟踪误差和失败率，再决定是否另行调整低等级起始容差。不得为过验收而静默改变映射。

## 2. 难度、位置和边界统一口径

- **奖励取当前位置**：用本次物理状态的 `torso_link` 世界 XY 调用 `terrain.tile_map.locate()`，读取 `family_group_index/tile_level/tile_valid`。torso 是当前跟踪 anchor；明确采用它，不含糊混用 pelvis/root、参考位置或最近 origin。纯 tile 元数据查询不需要新增高度射线。
- **课程取出生分配**：`terrain_levels/terrain_types` 和 curriculum 的 `birth_levels/birth_types` 继续用于出生、升降级及 outcome 统计。机器人从 A 跨到 B 后，奖励容差读取 B，课程仍结算给 A；本功能不改写参考统一变换。
- **等级与几何参数分开**：`tile_level=L0…L9` 用于本次容差的 L/9；manifest 的连续 `difficulty` 用于生成几何，作为诊断另存。当前 v6 的 L0/L9 几何 difficulty 约为 0.02405/0.95304，并非 0/1，禁止误将该字段直接代入已约定的 L/9 公式。
- **地形类型按语义匹配**：使用 `family_group_index` 对应的名称映射，避免 v6 的 family 列顺序与语义 group 编号混淆；旧 v5 的正向/倒置 stairs、slope 归入相应语义组。
- **小块与边界**：v6 一个 24×24 m 区域含 16 个 6×6 m 小块，区内样式和等级相同，跨小块容差不变。连接带仍属于其所在区域，不临时识别为 flat。双脚跨区域时，全身三项容差仍统一使用 torso 所在区域，T1 落脚几何继续逐脚查询。
- **跨区域立即按新区域评价**：沿用 `TerrainTileMap` 的边界归属，不增加平滑、滞回或“取两侧最大难度”。无有效 tile 时记录无效、容差取 0，不把 −1 索引或 NaN 当成有效地形，也不延用上一次容差。现有地图越界 truncation 保持，且仍检查实际/参考 tracked bodies，不能缩减为只检查 torso。

## 3. 分步实施与验收

| 顺序 | 工作和代码位置 | 完成标准 |
| --- | --- | --- |
| C1：固定配置与几何接口 | 新增 `pmt_tasks/pgmt_tracking_relaxation.py`，定义开关、三个 m、L/9、地形白名单、按名称的 α 和校验；复用 `pgmt_terrain_query.py`，明确当前几何/出生课程两套字段。 | 名称映射覆盖 12/12 个元素；配置拒绝负值、非有限值和未知名称；v5/v6 语义一致；不改课程分配。 |
| C2：实现纯数值容差 | 给 `pgmt_math.reward_terms()` 增加可选容差输入，只修改 lower 的 position/orientation/joint position 在指数核前的误差。保留默认严格路径，调试可同时返回原始/放宽误差与对应奖励率。 | 禁用、α=0、L0、flat/rough 回归原值；等于容差时残差为 0；超过容差只扣除规定 margin；上肢/速度/辅助/接触项不变。 |
| C3：接入每步缓存 | `PGMTState.rewards()` 从同一实际物理快照查询一次 tile；冻结元数据、margin、误差和旧回合结果。新增只读 `tracking_relaxation_state(for_reward=True)`，提供独立副本。 | 多个 RewardManager term 读取同一步不重复查询；终止步使用 reset 前 tile；局部 reset 只清当前行，保留旧奖励；参考不漂移、不修改动作。 |
| C4：独立验收 | 新增容差单测和有限步 Isaac 入口；复用原 21 项数值回归及 T1–T6 的 27 项/四组验收配置。 | 见下列覆盖矩阵；记录原始误差、容差、放宽误差和分项率，区分数值验证与策略完成率。 |
| C5：正式接入与续训契约（已完成） | 新 `pgmt_stage2_terrain` 默认开启三项容差、六项 terrain-contact 和四头；Stage1/原 P2 保留原默认。配置经 `builder.py` 传入，纳入实际环境契约和严格续训检查。 | 已验证保存作用项、坐标/等级来源、m、α、映射版本和解析后的名称顺序；修改配置拒绝续训；Stage1/P2 三头显式迁移，四头新进程恢复通过。 |

验收覆盖：

1. **CPU/CUDA 数值**：三个目标、五类地形、十级；小于/等于/大于 margin；α=0/逐元素覆写；保证 `0≤放宽误差≤原始误差` 且放宽奖励不低于严格奖励。验证无输入原地修改，原权重与 dt 各乘一次。
2. **原行为回归**：禁用时原 21 项逐值一致；启用时只允许三项 lower 改变。T1–T6、原始参考、command/观测保持一致。奖励数仍为独立配置 27 项，三项仍归 lower。
3. **几何和生命周期**：全部 50 个区域、旧 v5 类型映射、小块接缝、区域边界、出生 A/当前位置 B、不同环境局部 reset、越界和终止步；校验当前查询与冻结旧回合时间线。
4. **真实 Isaac**：有限环境逐级放置覆盖五类十级，独立重算三项误差/容差；再测跨 tile、reset、重复读及四组和。无效地图立即截断；记录每类每级的死区内比例与容差增益，避免只看总奖励变高就认定策略改善。

调试接口包含 `sample_step`、当前 family/level/geometry difficulty、tile_valid、出生 family/level，以及三项 `[N,12]` 的 raw_error/margin/relaxed_error 和 `[N]` 的 strict_rate/relaxed_rate。运行时冻结本步张量，通过显式接口读取独立副本，不自动输出逐环境日志；reset 后未结算行标为无效。

## 4. 与终止条件的衔接

本计划首先实现**奖励容差**。终止条件采用另一套误差，不能直接把 0.20 m/0.70 rad/0.50 rad 加到其阈值，更不能据此改变“完成动作才升级”的课程判据。

四头链路首轮短训沿用原终止标准；同日后续已按用户指定更新为 root XY >1 m、地形相对 torso 高度 >0.4 m、双手腕相对 pelvis 高度 >0.4 m、原姿态判据及 10 s 超时。后续按用户要求为前四项增加 0.2 s 连续超限容忍时间，各项/各环境独立计时，恢复正常或 reset 清零，无额外 reset 宽限；地图越界保持立即截断。公式与验收见 [Stage2 终止条件](PGMT_STAGE2_TERMINATIONS.zh-CN.md)，后续另评估地形适应性能。

## 5. 已实现的配置与调用接口

任务配置入口为 [`configs/task/pgmt_stage2_terrain.yaml`](../configs/task/pgmt_stage2_terrain.yaml) 的 `tracking_relaxation`，默认 `enabled: true`；环境配置对应 `cfg.pgmt_tracking_relaxation`。原 [`pgmt_stage2.yaml`](../configs/task/pgmt_stage2.yaml) 保持 `enabled: false`。新任务默认设置：

```yaml
tracking_relaxation:
  enabled: true
  position_max: 0.20       # m
  orientation_max: 0.70    # rad
  joint_max: 0.50          # rad
  terrain_families: [stairs, slope, boxes]
  level_max: 9             # scale = clip(current_tile_level / level_max, 0, 1)
  alpha:
    lower_body_position: {}
    lower_body_orientation: {}
    lower_joint_position: {}
```

`terrain_families` 可选 `flat/stairs/slope/boxes/rough` 的任意子集，空列表关闭所有地形的容差。使用语义名称，旧 v5 的正向/倒置样式自动按语义组归并。三项上限均要求有限非负；设 0 可单独关闭一项。`alpha` 按实际名称覆写，例如 `lower_joint_position: {left_knee_joint: 0.5, right_knee_joint: 0.5}`，未指定元素默认 1。未知字段/地形/部位、重复地形、负数或非有限配置会报错；上下限在环境创建前设置，不支持运行中直接改字典热更新。

Python 构建接口：

```python
cfg = build_env_cfg("pgmt_stage2", overrides=[
    "tracking_relaxation.enabled=true",
    "tracking_relaxation.position_max=0.12",
    "tracking_relaxation.terrain_families=[stairs,boxes]",
])
```

现有 `scripts/train_pgmt_stage2.py` 增加 `--tracking-relaxation` 显式开启选项，以及 `--tracking-relaxation-config /path/to/relaxation.yaml`。后者读取**不带外层 `tracking_relaxation:`** 的参数映射；文件中设置 `enabled: true`，或同时传开启选项。两者同时出现时，开启选项优先。实际配置写入 `params/env.yaml` 和 training contract；启用时课程 checkpoint identity 也记录版本、坐标/等级来源、解析后的 link/joint 顺序和参数，修改配置不能按同任务严格续训。

代码位置：

- [`pgmt_tracking_relaxation.py`](../pmt_tasks/pgmt_tracking_relaxation.py)：默认值、校验、12/12 名称解析、类型/等级缩放、非负截断。
- [`pgmt_math.reward_terms()`](../pmt_tasks/pgmt_math.py)：三项逐元素误差进入指数核前扣减容差；其余 18 项与原来相同。
- [`PGMTState.rewards()` / `tracking_relaxation_state()`](../pmt_tasks/pgmt_runtime.py)：本步实际 torso 查询、终止步冻结、局部 reset 失效和只读副本。

调试示例（先由 RewardManager 结算本控制步）：

```python
sample = get_pgmt_state(env).tracking_relaxation_state(for_reward=True)
raw = sample["lower_body_position_raw_error"]         # [N,12]
margin = sample["lower_body_position_margin"]         # [N,12]
error = sample["lower_body_position_relaxed_error"]    # [N,12], clamp(raw-margin, min=0)
```

另有 `lower_body_orientation_*`、`lower_joint_position_*`、三项 `*_strict_rate/*_relaxed_rate`，以及 `anchor_xy_w/tile_level/family_group_index/difficulty/birth_level/birth_family_index/reference_motion_ids/reference_frame_ids/sample_step`。`sample_valid` 表示该行有已结算样本；`tile_valid` 表示地图查询有效，二者不能混用。接口默认 `for_reward=False` 读取 live 样本，reset 行无效；`for_reward=True` 保留终止步旧样本，若本步尚未结算则报错。可传 `env_ids` 限定环境，返回值修改不会污染缓存。

## 6. 2026-10-14 容差独立验收记录（先于四头接入）

- **CPU/CUDA**：容差与 T1–T6、原 MDP、配置、地形查询、参考、课程和 checkpoint 的联合回归 **267 passed**；之后补充配置身份变化拒绝续训测试，容差专项 **30 passed**。覆盖 v5 七样式 × 十级、v6 五样式 × 十级、可配置地形/上限/α、名称乱序、等于和超过容差、无效地图、非有限原始误差、跨小块/跨区域、重复读取和局部 reset。禁用/零容差保持严格数值，放宽后 error 非负且奖励不低于严格值。
- **真实 Isaac**：[`isaac_01/training_status.json`](../logs/pgmt_p3_tracking_relaxation/isaac_01/training_status.json) 为 `passed`。5 环境、14 个控制步，实际随机初始化并步进覆盖 **50 个区域**；独立 NumPy 旋转矩阵重算原始误差，最大差异 **6.20196487e−7**。额外移位检查出生 stairs L9 / 当前 slope L3、参考平移不变、局部终止/reset，以及原生地图越界 truncation。
- **RewardManager**：独立配置共 27 项 / 四个奖励分组；仅三项 lower 跟踪改变，其余原 18 项逐值一致，分项权重/dt 一次与分组总和通过。未构造四头策略、未进行 PPO 更新，不能据此认定 P3.4 已完成。
- 原始误差、容差内比例和奖励增益记录在 [`measurements.json`](../logs/pgmt_p3_tracking_relaxation/isaac_01/measurements.json)，参数与部位顺序记录在 [`relaxation_contract.json`](../logs/pgmt_p3_tracking_relaxation/isaac_01/relaxation_contract.json)。这些是数值和生命周期验收，不是长期跟踪成功率验证。
- Isaac 验收进程已自动退出；没有启动训练。

复现入口：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/lzp/miniconda3/envs/isaacsim/bin/python -m pytest tests/test_pgmt_tracking_relaxation.py -q
bash scripts/validate_pgmt_tracking_relaxation.sh --log_dir logs/pgmt_p3_tracking_relaxation/isaac_next
```

Isaac 使用既有 native robot USD、walk1 动作和 v6 地形。为隔离奖励与 reset，验收只保留地图越界终止和受控终止；真实训练的原终止规则未修改。跨区域探针通过移动实际 root 构造，不能表述为策略自主跨越地形成功。

## 7. C5 正式训练与续训验收

同日后续已在完整 862 动作/160 环境正式任务中默认启用容差，完成 3+2+1 次 PPO 更新和多次独立进程恢复。四头/奖励/容差/感知联合回归 **415 passed**；保存的实际环境配置和课程 identity 均包含容差。修改上限等契约后严格续训被拒绝，旧三头通过显式新阶段迁移进入四头。

容差仍只改变三项 lower 的误差，未修改 command、原始关节轨迹或出生课程判据。过程与结果见 [四头验收文档](PGMT_P34_FOUR_HEAD_ACCEPTANCE.zh-CN.md) 和 [独立报告](../logs/pgmt_p3_four_head/acceptance_report.json)。该首轮报告对应旧终止标准；后续用户指定的新终止方案单独验收和记录，分级性能评估待完成。
