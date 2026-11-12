# PGMT 参考放置与初始化

已实现 P1 的统一参考变换，并接入真实 Isaac Lab 环境。使用现有 `pgmt_stage2_v5` 地形及带 TCRS 接触标签的 50 Hz NPZ。2026-10-10 已接入动作级自适应及用户提供的动作—地形兼容表。此配置用于放置和初始化验收；感知、stage 2 奖励和恢复策略仍按总计划继续构建。

## 2026-10-15：Stage2 仅从中央 2×2 小地形出生

按用户最终确定的 **2×2** 实现。v6 每个“地形样式 × 难度”区域仍为 24×24 m，包含 4×4 个 6×6 m 小块；两个 Stage2 任务仅从中心 **12×12 m** 范围内的四个完整小块初始化，编号为 **5、6、9、10**（行列从 0 开始）。外围一圈保留，地形资产不重新生成。

```text
 0   1   2   3
 4  [5] [6]  7
 8  [9][10] 11
12  13  14  15
```

先在允许的四块中等概率抽样，再在选中小块内部均匀采样 XY；沿用每块 0.25 m 边距、全身防穿透/边界检查和不合格姿态重试。中央区外沿距所属区域边界为 6 m，根位置候选再留 0.25 m 余量。yaw 仍从 `[-π,π]` 抽样；有效姿态的最终分布还受几何拒绝检查影响。

配置位于 [stage2.py](../pmt_tasks/env_cfgs/pgmt/stage2.py)：

```python
self.commands.motion.placement_patch_grid = (2, 2)
```

`PGMTTerrainMotionCommandCfg.placement_patch_grid` 默认为 `None`，保留历史 P1 全 4×4 采样与旧 v5 origin 放置；Stage2 显式启用 `(2,2)`。在环境最终配置完成后、创建环境前设为 `None` 或 `(4,4)`，可恢复 v6 全范围采样。设置要求选取对称居中的完整小块，4×4 上的 `(3,3)` 会明确报错，避免隐含偏向一侧。

- [pgmt_terrain_layout.py](../pmt_tasks/pgmt_terrain_layout.py)：计算中心小块编号及限定候选采样。
- [pgmt_motion_command.py](../pmt_tasks/mdp/commands/pgmt_motion_command.py)：启动时缓存允许编号，首次、局部、自动及课程更新后的 reset 共用该路径，优化放置路径也调用此处。
- [pgmt_curriculum.py](../pmt_tasks/pgmt_curriculum.py)：非默认出生网格写入 checkpoint 课程 identity；正式环境配置也进入训练契约。旧 checkpoint 不会被静默视为出生分布相同，文件未修改。

这只约束出生位置。运行时仍可进入外围小块、跨 tile，地形查询继续按实际位置；超出整张有效地图才触发既有越界截断。动作自适应采样、禁配表、起始第 0 帧、统一参考变换与源关节轨迹沿用原逻辑。本次没有用训练数据估计越界率的降幅。

验收：CPU **57 passed**；CPU/CUDA **58 passed**，包含每个设备 20,000 个候选，覆盖全部 50 个样式/等级区域、中央四块、局部/空 reset、原全 4×4 分支、外围地图有效性及课程/参考回归。真实 `pmt / Isaac Sim 5.1`、GPU 1、headless、5 环境有限验收通过：记录的 100 个出生点覆盖五类十级，四块计数为 21/23/28/28，最大中心 XY 偏移 5.546 m，yaw 范围约 −3.025～3.131 rad，最大放置重试 5 次；合计检查 121 次实际 reset、56 控制步，21 次 done 对应 21 次 sampler outcome。使用一个 walk 动作做受控物理探针，不替代历史全动作库穷举验收。仿真已关闭，未启动 PPO 训练。

结果：[acceptance_report.json](../logs/pgmt_center_spawn_20261015/acceptance_report.json)；原始位置/yaw：[measurements.json](../logs/pgmt_center_spawn_20261015/isaac_01/measurements.json)。

## 坐标约定

每次 reset 以**实际选中的起始帧**建立一次固定变换。按用户当前要求，默认固定为第 0 帧；检查工具仍可显式指定其他帧：

```text
R = yaw(目标世界朝向 - 起始帧 pelvis 朝向)
t_xy = 目标 tile 内出生点 xy - (R * 起始 pelvis 位置)_xy
p_world = R * p_source + t
q_world = q_R ⊗ q_source
v_world = R * v_source
ω_world = R * ω_source
```

`t_z` 来自真实地形网格上的放置查询。整个 episode 使用同一组 `R,t`，不会每帧重新贴地；关节位置、关节速度和动作内部的相对轨迹保持原样。倒金字塔的负高度不会截为 0。14 个 command 跟踪刚体、30 个全身参考刚体、8 个未来帧以及机器人 reset 共用该变换，不再重复叠加 `env_origins`。

未来帧仍采用 `[0,1,3,7,15,31,63,127]`，超出 segment 时位姿取末帧。`PGMTState.reference_contacts()` 与位姿共用 motion/frame 索引；超界未来接触标签的有效性设为 False，避免重复末帧形成错误监督，详见 [训练期接触接口](PGMT_CONTACT_LABELS.zh-CN.md)。root reset 使用 **link 位姿/速度**接口，避免把参考 link 速度当成质心速度；写入后更新运动学，再计算 command 相对参考。

## 地形和边界检查

- 读取固定地形 manifest，核验 USD 和 70 个缓存 mesh 的 SHA256；GPU 查询使用相同世界坐标下的三角网格。
- 向下射线取得地面高度；再用训练 G1 URDF 的全身碰撞形状保守球覆盖，检查到三角面的最近距离，包括斜面和楼梯立面。初始化预留约 3 mm 间隙。这里使用训练机器人碰撞模型，与离线 TCRS 四球标签模型不同。
- 检查起始碰撞覆盖处于出生 tile 内，留 0.25 m 余量；初始 8 帧参考窗口允许进入相邻 tile，但必须位于整张有效地图内。默认在 tile 中央 ±0.25 m 内采样 xy，世界 yaw 在 `[-π,π]` 内采样。无射线命中或不合格的放置重试，32 次仍失败则明确报错，不裁剪关节或强行放置。
- 当前暂时关闭起始帧随机化及足部有效支撑限制，直接使用第 0 帧；`contacts`、`contact_valid` 不参与 reset 候选过滤，爬行、足部悬空或标签无效不会因此被排除。地形兼容表、帧索引范围、关节限位、网格防穿透和边界检查仍然生效。此调整没有实现完整的跌倒恢复策略。
- 运行期允许当前参考和机器人跨越出生 tile 的边界；任一当前跟踪刚体的参考位置或机器人位置离开**有效地图**，立即触发 `map_out_of_bounds` truncation（`time_out=True`），不等待失败延迟。动作耗尽也通过 episode 结束处理，不在运行中直接瞬移换片。

**范围说明：** 无穿透检查针对初始化姿态；固定刚体变换不会让后续平地动作自动适应楼梯。后续动作与地形的适配仍由感知、奖励和策略训练处理。本次检查使用 `dance1_subject1_clip0.npz`，不表示整个动作库的所有帧都已通过地形兼容性验收。

## 跨 tile 查询与课程归属（2026-10-10）

地形查询以传入的**当前世界 XY 坐标**定位实际 tile，左右脚可以同时查到不同类型或不同等级。`terrain.query(xy)` 接受 `[...,2]` 坐标，返回同形状的 `height`、`family_index`、`family_group_index`、`tile_level`、`difficulty`、`tile_valid` 和 `valid`。其中 `tile_level`/`difficulty` 描述当前位置的真实几何；`up_stairs`/`up_slope` 的语义组分别映射为 stairs/slopes。高度直接查询全局真实三角网格，保留负高度。

**课程等级仍属于出生分配。** `terrain.terrain_levels`、`terrain.terrain_types` 和 `env_origins` 继续保存分配信息，位置查询不会改写它们；跨 tile 不改变出生等级、不触发升级或重新抽动作。地形课程已接入：完整动作跟踪升一级，跟踪失败降一级，只在回合结算/reset 时更新等级和出生位置，L0–L9 封顶；地图越界和提前超时保持等级。详见 [独立课程判据、统计与续训](PGMT_TERRAIN_CURRICULUM.zh-CN.md)。2026-10-14 的[容差实施计划](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)已将奖励容差的 family/level 明确为当前实际 torso 所在 tile，出生等级仅用于课程记账；该奖励修改尚未实现，失败延迟另行规划。动作兼容表继续用于 reset 时的出生地形候选筛选。

后续 reward 可直接使用：

```python
from pmt_tasks.pgmt_runtime import get_pgmt_state

ground = get_pgmt_state(env).robot_foot_terrain()
# ground["height"], ["family_index"], ["tile_level"], ["valid"]: [N,2]，左/右脚
# ground["curriculum_level"], ["assigned_family_index"]: [N]，出生分配
```

有效地图为 70 块训练 tile 的覆盖范围，当前 v5 为 **X∈[-120,120)、Y∈[-84,84) m**；外围 20 m border 不计入。内部边界采用半开区间，边界点归正方向的相邻 tile，不在内部 tile 边缘施加 reset 的 0.25 m 余量。地图外坐标返回 `tile_valid=False`、索引 −1、高度 NaN；图内射线缺失时 `valid=False`、高度 NaN，不填零或夹到出生 tile。扫描窗口中个别点越界只标无效；回合 truncation 由当前参考/机器人跟踪刚体是否越界决定，不由未来窗口或扫描点直接触发。

验证入口：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt_terrain_query.py
/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_map_queries.py \
  --headless --device cuda:0 --output logs/pgmt_map_queries
```

46 项相关单元/回归测试通过。Isaac [验证报告](../logs/pgmt_map_queries/validation.json) 已通过：70 块地形的 720 个审计探针，高度最大误差约 4.1×10⁻⁶ m；跨等级、跨类型和双脚跨缝查询不改变出生分配；参考和机器人分别越过地图边界时均产生 truncation。脚本为隔离地图退出语义，临时关闭其他跟踪终止项；生产配置保留原有终止项。

## 动作级自适应与地形兼容（2026-10-10）

以下描述保留历史 P1 配置。2026-11-11 起，Stage2 已切换为动作级 EMD 权重与片段内均匀随机起点，见 [EMD 采样说明](PGMT_EMD_SAMPLING.zh-CN.md)。

历史 P1 放置配置采用 `sampler_type="adaptive"`、`enable_frame_sampling=False`、`chunk_length=0`，设置 `fixed_start_frame=0`、`require_valid_support_start=False`。一个 NPZ segment 对应一个动作采样单位，复用 PMT `AdaptiveSampler` 的动作级失败/成功计数、EMA、beta 权重及均匀探索比例；不建立帧级或 bin 级失败统计。Stage1 原配置保持不变。

启动时读取 `assets/motions/pgmt_clipped_50hz_tcrs_v1/motion_terrain_compatibility.csv`，按 `motion_name` 对应实际 loader 中的 NPZ 文件名，记录 CSV SHA256。`dismatched terrain` 为空表示无地形禁用；`stairs` 同时排除 `stairs` 和 `up_stairs`，`slopes` 同时排除 `slopes` 和 `up_slope`。也可以单独指定倒金字塔变体；多个地形名用分号、空格或带 CSV 引号的逗号分隔。缺少选中动作、重复名称、未知地形会报错，避免拼写错误导致过滤失效。修改表后重新创建环境即可重新加载。

每次 reset 按该环境当前分配的地形列确定允许的动作，只在这个集合内归一化 `p_fail ** beta` 和均匀分量，禁用动作的最终概率严格为零。选定动作后直接取第 0 帧，不再抽取随机起始帧。帧级自适应开关与起始帧选择是两个独立设置，本次两者均关闭。检查脚本的 `--start-frame` 可显式覆盖固定帧，但不会绕过兼容表或真实帧范围。后续如需恢复随机有效支撑起点，可将 `fixed_start_frame` 设为 `-1`、`require_valid_support_start` 设为 `True`。

当前表有 862 行，其中 359 个片段禁用楼梯。`walk3_subject3` 的 `clip18`、`clip22`、`clip23` 是膝部支撑的爬行片段，不能仅因没有足部支撑而判为无效；现在这三段重新进入兼容地形的候选集合。本次不修改用户指出的重定向问题。当前普通楼梯、上楼梯各有 **503** 个采样候选，其余五种地形各有 **862** 个候选；是否能完成具体姿态放置仍由几何检查判断。

兼容过滤初版验证记录保存在 `logs/pgmt_sampling/validation.json`，该记录对应此前的随机支撑起始帧模式。当前固定第 0 帧的验证另存于 `logs/pgmt_sampling_frame0/validation.json`，检查采样帧全为 0、足部支撑限制关闭、兼容过滤和动作级统计继续生效，并单独检查三个爬行片段的平地 reset。此短程检查不代表全库全部地形组合的姿态放置验收已完成。

```bash
/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_sampling.py \
  --headless --device cuda:0 --output logs/pgmt_sampling_frame0
```

## 实现入口

| 文件 | 职责 |
| --- | --- |
| `pmt_tasks/pgmt_reference.py` | 统一变换、训练碰撞形状覆盖、网格放置及边界检查 |
| `pmt_tasks/pgmt_terrain_importer.py` | v5 地形导入、世界高度和最近三角面查询 |
| `pmt_tasks/pgmt_terrain_query.py` | 按世界 XY 定位实际 tile、语义组/几何等级和有效地图边界 |
| `pmt_tasks/pgmt_curriculum.py` | 独立课程成功判据、出生等级更新、课程统计与 checkpoint 状态 |
| `pmt_tasks/pgmt_sampling.py` | 兼容表解析、地形内动作概率及固定/随机起始帧选择 |
| `pmt_tasks/pgmt_contact_reference.py` | 设备端接触标签打包、左右脚对齐、当前/未来帧索引与有效性 |
| `pmt_tasks/mdp/commands/pgmt_motion_command.py` | segment/failure-adaptive sampler 复用、参考缓存、共享 reset、耗尽/越界截断 |
| `pmt_tasks/pgmt_runtime.py` | 当前/未来全身参考、接触目标、实测足力；`robot_foot_terrain()` 查询当前脚下地形并单独返回出生课程等级；stage 1 保留原有路径 |
| `pmt_tasks/env_cfgs/pgmt/placement.py` | P1 初始化环境配置 |
| `scripts/inspect_pgmt_placement.py` | 真实 Isaac 环境验证和交互检查窗口 |
| `scripts/view_pgmt_placement.py` | 在 Isaac 6 中显示实际 PMT reset 状态，兼容本机的桌面渲染环境 |

## 验证与查看

2026-10-09：15 项相关单元/回归测试通过。第 0 和 249 起始帧分别检查七种地形 × 十级 × 三种 yaw，共 420 组初始化；PhysX 未检测到地形重叠。机器人与全身参考位置最大误差约 0.023 mm。每次验收先做 650 个地形探针，PhysX 与缓存网格高度误差最大约 0.0041 mm，再执行碰撞检查；headless 模式显式启用场景查询，避免“查询未启用”的空结果被误判为无碰撞。另验证部分 reset 不影响其他环境、当前/未来参考一致和片段耗尽不瞬移。

验证报告：`logs/pgmt_placement/validation.json`、`logs/pgmt_placement_frame249/validation.json`。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/inspect_pgmt_placement.py \
  --headless --validate --device cuda:0
```

加 `--start-frame 249 --output logs/pgmt_placement_frame249` 可检查非零起始帧。窗口支持切换七种地形、L0–L9 和 yaw；保持 reset 姿态，便于检查出生位置。窗口不执行策略推理或训练。

本机 Isaac 5.1 的 GUI 在 RTX 渲染器启动阶段崩溃，物理环境 headless 运行正常。采用以下两步在已可用的 Isaac 6 中检查：先由 PMT 环境完成真实 reset 并导出 USD、30 个刚体的实际世界位姿，再打开检查窗口。窗口会重新核验 USD 世界变换与导出的 PhysX 状态一致。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/inspect_pgmt_placement.py \
  --headless --validate --export-scene --device cuda:0
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/view_pgmt_placement.py \
  --snapshot logs/pgmt_placement --gpu 1 --family up_stairs --level 5 --yaw-deg 90
```

2026-10-10 起，检查脚本默认动作改为兼容全部地形的 `walk1_subject1_clip0.npz`。旧的 dance 初始化导出来自兼容表接入前，应执行上述第一条 Python 命令重新生成，再用第二条打开。默认显示上楼梯 L5、yaw 90°、动作第 0 帧。`PGMT Initialization` 面板可切换地形、等级、已验证的 yaw（0°、90°、−135°）及近景/整块地形视角。显示的是初始化快照，不是 Isaac 6 中运行的 PMT 训练或控制策略。若指定仅含 dance 的动作集合并请求七种地形检查，楼梯候选为空时会明确报错，不会绕过兼容表。
