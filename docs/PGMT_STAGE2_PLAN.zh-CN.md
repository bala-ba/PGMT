# PGMT 第二阶段构建计划

日期：2026-10-08；P2 状态更新：2026-10-13；P3/P4 状态更新：2026-10-15（四头正式任务、扩容/续训通过，8192 环境完成 383 更新后保存停止，recovery 暂缓）。依据：本项目 [PGMT 论文](../PGMT_LITERATURE.pdf) 第 III–IV 节、图 2、表 I，以及当前 PMT 源码和实际 checkpoint。

**推荐路线：复用 PMT 训练框架和现有资产，依次完成「基线固定 → 地形和重置 → 感知注入 → 奖励与课程 → 验证和训练」。** 沿用原始动作库，独立采样地形，通过 PPO 联合训练，无须生成地形配对动作或引入蒸馏。

当前目标是完成不含 recovery 的 PGMT 地形跟踪训练闭环。**按用户 2026-10-14 的要求，暂不构建或接入 recovery pool，不实现 recovery upward velocity 及配套恢复课程，也不将其列为本轮 P3/P4 验收前置条件。** 与论文的这一差异保留记录，相关方案暂缓、未计为完成，保留为后续可选任务。未公开的地形参数、奖励细节、PPO 超参数和总迭代数记录为工程假设。

**P3 总验收更新（2026-10-14）：部分通过，尚未全部完成。** P3.1–P3.4 核心实现和最新配置链路通过；600 项 CPU/CUDA 回归、160 环境 3+2 次 PPO 更新、两次新进程恢复及 Stage1/P2 迁移通过，覆盖双腕 EE 奖励、四项 0.2 s 容忍和 10 s 回合。P3.5 已补齐头/胸真实碰撞→sensor→奖励→reset 验证（5 环境/44 步），发现 200 Hz 物理与 50 Hz 末子步奖励采样之间的短冲击峰值遗漏；采样聚合尚未修改，爬行接触冲突统计仍待完成。P3.6 尚缺充分的分地形/难度性能统计。详见 [身体碰撞验收](PGMT_BODY_CONTACT_AUDIT.zh-CN.md) 与 [P3 总验收与缺口](PGMT_P3_ACCEPTANCE.zh-CN.md)。

## 1. 已有资源与必须补齐的差异

| 项目 | 当前状态与处理 |
| --- | --- |
| 指定权重 | `logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt`；已核实 iteration、三头结构及 tensor 有限性。SHA-256：`40708024acb51c6ce3cd87d6a067f59ff822763b6d8b73d3e2473d9953793087`。 |
| 控制与网络 | 保留 29 DoF G1、PMT PD/动作缩放、50 Hz 控制，及 `96 / 10×96 / 8×61 / 251` 观测契约；未来帧 `0,1,3,7,15,31,63,127`，embedding 128、4 attention heads。 |
| stage 1 动作与 PPO | 40 条 LAFAN1、441,140 帧、136 个 chunk；保留三头基线，P3 已扩展四头、按组 GAE、优势先求和后归一化。 |
| stage 2 动作预处理 | `clipped_csv` 当前 862 个片段已转换为 50 Hz PMT NPZ（430,138 帧），附 TCRS/网格接触标签及有效性；输出为 `assets/motions/pgmt_clipped_50hz_tcrs_v1`。动作级自适应、CSV 地形兼容过滤和训练期接触标签已接入正式四头任务 `pgmt_stage2_terrain`。 |
| 可迁移资源 | `../PGMT/pgmt/` 的 glimpse/地形/训练辅助实现及 `../PGMT/assets/terrain_cache/independent_v1` 五类十级缓存，均为本地独立实现，校验后复用。已有 `recovery_pool.pt` 的 4,096 个状态仅保留资源记录，本轮不接入。 |
| 后续缺口 | 地形奖励/容差、四头及有限训练/续训已完成；Stage2 已按用户指定改为地形相对 torso、相对 pelvis 手腕、root XY 和 10 s 终止。身体碰撞链路及 1024 PPO/恢复通过，8192 环境完成 383 更新；短冲击采样、动作接触语义与长期性能评测仍待处理，倒地池恢复暂缓。Stage1/P1 原 15 s、当前 Stage2 10 s 与最终 30 s 评测分别记录。 |

已有 [model_15000 平地报告](../logs/pgmt_stage1_eval/model_15000_flat_20260929/REPORT.md) 可作迁移回归依据，但其随机片段、15 s 重置统计不是论文的连续 30 s 完成率。

## 2. 实施顺序与验收

### P0：固定接口、数据和迁移规则

以下跨阶段迁移约定已由 P2/P3.4 实现；不追溯计为 P1 工作。新增 `pgmt_stage2_terrain` / `PMT-PGMT-Stage2-Terrain-G1-v0` 和 `train_pgmt_terrain.sh`，保留原 `pgmt_stage2` 三头感知基线，见 [四头验收](PGMT_P34_FOUR_HEAD_ACCEPTANCE.zh-CN.md)。

- 新建 `pgmt_stage2` 配置及 `PMT-PGMT-Stage2-G1-v0`，保留 stage 1 默认行为；保存 checkpoint、`params/`、数据/机器人/源码指纹，论文事实以本次 PDF 为准。
- 区分**跨阶段初始化**与**同阶段续训**：从 `model_15000.pt` 复制主干、actor、已有 critic、探索标准差及归一化统计；新建优化器，stage 2 从 iteration 0 计数。
- critic 顺序为 `[upper, lower, terrain, auxiliary]`：现有 `critic.6` 第 0/1/2 行映射到第 0/1/3 行，terrain 行零初始化；逐项核验迁移。

**验收：** 关闭 terrain token 的 attention 后，新旧网络在同一观测上的确定性动作和三个旧价值输出一致；新 head 与元数据正确。仅输入零地图不保证等价，因为新增 K/V 会改变 softmax 分母。

### P1：先完成地形、参考放置和接触数据

当前采用可选的 **pgmt_stage2_v6**：五类原生地形、十级、50 个 24×24 m 区域，每区域含 16 个 6×6 m 小块，共 800 块；随机出生 XY/yaw，动作仍从第 0 帧开始。旧 v5 七类地形及其独立验收保留，P1 入口默认 v5，显式 `--terrain_asset assets/terrain/pgmt_stage2_v6` 选择新版；P2 感知入口默认 v6。见 [v6 地形说明](PGMT_TERRAIN_V6_PREVIEW.zh-CN.md) 与 [1000 次无感知 PPO 更新记录](PGMT_V6_BLIND_TEST.zh-CN.md)。五类按列分配，当前 160 个环境各 32 个，已实现各 20% 的环境分配。统一参考、接触数据、兼容过滤、跨区域查询及独立课程均已接入。

- [x] **动作数据人工筛选**：由用户负责，按用户要求在计划中标记完成（2026-10-09）。数据源为 `/home/lzp/projects/perceptive_tracking/data/lafan_prepared/clipped_csv`，后续接入以人工筛选后的数据清单为准。格式转换已随离线接触标签预处理完成；2026-10-10 已接入以每个 NPZ segment 为单位的动作级自适应和用户 CSV 兼容过滤。

- [x] 使用 Isaac Lab 原生接口构建 **flat / stairs / slope / boxes / rough × L0–L9**；stairs/slope 均为倒金字塔，随机 XY/yaw 表达上行和下行。保存网格、区域/小块索引、等级、原点、参数、seed 和哈希。v6 名称映射既有 CSV 的 `flat / up_stairs / up_slope / boxes / random_rough`，兼容原语义组排除规则；旧资产保留。
- [x] 新增 PGMT command：采用 `sampler_type="adaptive"`、`enable_frame_sampling=False`、`chunk_length=0`；读取 `assets/motions/pgmt_clipped_50hz_tcrs_v1/motion_terrain_compatibility.csv` 排除不兼容组合。动作失败统计按 segment 汇总，先确定地形的兼容集合，再归一化自适应权重并混合均匀采样。暂设 `fixed_start_frame=0`、`require_valid_support_start=False`，关闭随机起始帧和足部支撑过滤；3 个膝部支撑爬行片段保留在候选集合。v6 使用五类列等量分配；旧模式报告 `logs/pgmt_sampling_frame0/validation.json` 保留。
- [x] 以实际起始帧建立统一平移/yaw 变换，供 command、全身参考、未来帧和 reset 共用；查询真实地形高度及全身碰撞形状的网格距离，检查初始化穿透与边界，保留原始关节轨迹。`PGMTState.reference()` 已接入共享变换，stage 1 保留原有行为。已验证第 0/249 起始帧共 420 组地形/等级/yaw 放置；实现与检查入口见 [参考放置与初始化](PGMT_REFERENCE_PLACEMENT.zh-CN.md)。当前固定第 0 帧模式另已完成全库 862 个动作、159,480 个候选放置检查；两个不安全候选均被正确拒绝，并通过真实 command 重试安全初始化，未删除动作或修改轨迹。
- [x] 倒金字塔保持中央低、四周高及真实负高度。v6 按用户要求在小块内随机出生，不再限定中央低平台；出生区域 origin 用于课程分配，机器人 Z 由实际 XY 的地形与全身碰撞决定。torso 中心、yaw 对齐的 21×21 原生 RayCaster 共享完整查询网格；保留带符号高度和无效掩码，不固定为 +X。旧 v5 中央放置/扫描证据保留，见 [扫描接口与验证](PGMT_HEIGHT_SCAN.zh-CN.md)。
- [x] reset/奖励使用干净高度查询；`terrain.query(xy)` / `PGMTState.robot_foot_terrain()` 按当前世界坐标查询实际地形类型、高度和几何等级，双脚可分别落在不同 tile。**课程等级仍使用出生分配**，跨 tile 不改写 `terrain_levels/types` 或 `env_origins`；P3 奖励容差另按[容差实施计划](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)读取当前位置的 family/level，失败延迟单独规划。允许跨 tile，当前参考/机器人跟踪刚体超出有效地图（不含外围 border）时立即 `time_out=True` truncation。扫描越界点单独标无效，不当作零高度。详见 [参考放置与跨 tile 查询](PGMT_REFERENCE_PLACEMENT.zh-CN.md)。课程更新已接入，见 [地形课程与续训](PGMT_TERRAIN_CURRICULUM.zh-CN.md)。
- [x] 离线网格查询接触标签已构建：复用 TCRS 的高度/速度判定及连续段去抖，补充参考网格射线和 G1 足底碰撞球检查；当前 CSV 生成 862 个 50 Hz segment、430,138 帧，保存 `contacts`、`contact_valid`、左右脚顺序和源帧映射。见 [接触标签说明](PGMT_CONTACT_LABELS.zh-CN.md)。该阈值协议记录为工程选择。
- [x] 训练期接触标签接口：`PGMTState.reference_contacts()` 按 command 当前 motion/frame 返回左右脚目标、有效性和支撑/摆动 mask，支持未来 8 帧和局部环境查询；片段末尾之外的未来帧屏蔽监督。`robot_foot_contact_forces()` 返回相同顺序的实际接触力。接口独立于 reset 足部支撑过滤，兼容固定第 0 帧及爬行动作；每次查询跟随最新索引，无跨 reset 旧标签缓存。38 项单元/回归测试及 Isaac 全库标签、reset/reward 时序检查通过；验证入口及使用约定见 [接触标签说明](PGMT_CONTACT_LABELS.zh-CN.md)。P3 已补齐实际接触事件状态、六项接触奖励及正式四头训练接入。
- [x] 动作耗尽显式截断（当前 `chunk_length=0`，按完整 segment），先结算旧奖励和旧 sampler outcome 再重置机器人，done 切断 GAE。已接入独立课程成功判据：从第 0 帧完成整段跟踪且无失败/地图越界才升级，跟踪失败降级，其他中性截断保持等级；单独出现末帧标志不足以升级。reset 更新出生等级与 origin，首次/手动/恢复 reset 不计 outcome。课程与采样统计已随 checkpoint 保存恢复，详见 [地形课程与续训](PGMT_TERRAIN_CURRICULUM.zh-CN.md)。

**当前验收：P1 已完成（2026-10-11，v6、固定第 0 帧）。** 既有五类十级 mesh/碰撞/高度查询通过；本次补齐 1,264,320 个随机候选、39,510 次全库兼容组合实际 reset、800 小块独立 PhysX 重叠检查、3,200 个扫描位姿与实际 checkpoint 续训/二次保存恢复。119 项回归测试通过，常用训练入口加载 v6 课程统计并继续 PPO 成功。完整结果、修复及范围见 [v6 P1 补充验收](PGMT_V6_P1_ACCEPTANCE.zh-CN.md)，总报告 `logs/pgmt_v6_acceptance/acceptance_20261011.json`。

随机动作起始帧按用户要求保持关闭，恢复该模式后再扩展对应验收。2026-10-10 的旧 v5 七类验收及 998 步生命周期报告仍独立保留于 `logs/pgmt_p1_acceptance/single_scene/acceptance.json`，见 [旧版验收记录](PGMT_P1_ACCEPTANCE.zh-CN.md)。

#### 当前 v6 difficulty range 与布局

五类十级，生成时 `d=(L+u)/10`，固定 seed 后不在 reset 时重建。每区域 24×24 m，4×4 个 6×6 m 小块；初始 L0–L1。参数以 `configs/terrain/pgmt_stage2_v6.yaml` 和已校验的 v6 manifest 为准。

| 样式 | 难度范围及当前设置 |
| --- | --- |
| flat | 原生平面，十级均为平地 |
| stairs | 原生倒金字塔；阶高 0.05–0.30 m；踏深固定 0.32 m，暂时关闭深度变化 |
| slope | 原生倒金字塔；坡角 0–30°；中央平台参数 0.5 m |
| boxes | 原生 random grid，幅度 H=0.05–0.40 m，顶面随机取 [-H,+H]；箱宽 0.7 m；无中央平台；相邻小块连接总宽 0.4 m |
| rough | 原生 random uniform；输入噪声半幅 1–8 cm；保留原生插值与陡面修正 |

v6 boxes 是用户确认的原生正负扰动，不能用下面 v5 历史的正高度箱体规则解释。随机位置检查覆盖起始姿态；P2 感知融合已接入，地形接触奖励仍在 P3。

#### 历史 v5 参数（保留用于回退，不作为 v6 配置）

论文规定五类地形、十级课程，但未公开各类几何范围。以下为 **PMT 首版工程设定**：斜坡、楼梯和箱体采用用户指定范围，以低难度承接平地 checkpoint；保留 rough 范围并修正随机箱体的正负高度语义。PMT 现有 `ROUGH_TERRAIN_CFG` 中 5–10 cm 台阶仅是比例为零的候选配置，不代表 stage 1 已训练过这些地形。

```yaml
# 已接入 P1 无感知地形训练；完整 stage 2 感知/奖励闭环待完成。
curriculum: true
difficulty_range: [0.0, 1.0]
num_rows: 10
num_cols: 7                  # 五类原有地形 + 两种倒金字塔变体；采样比例另设
max_init_terrain_level: 1    # 初始仅 L0–L1，完整地形范围仍为 [0,1]
size: [24.0, 24.0]
horizontal_scale: 0.10       # slopes 网格；rough 单独采用 PGMT 的 0.15 m；height scan 另设 0.10 m
vertical_scale: 0.005
```

| 地形 | 随 difficulty 增长的范围 | 固定几何/实现约定 |
| --- | --- | --- |
| flat | 高度变化 0 | 十级均保持平地，用于保留跟踪能力 |
| slopes | 坡角 `theta=30d` 度，范围 **0°–30°**；生成器坡度系数 `s=tan(pi*d/6)`，范围 **0–0.577350** | 按角度线性增加；覆盖上/下坡，中心平台宽 2 m |
| stairs | 单级高度 `h=0.05+0.25d` m，范围 **0.05–0.30 m**；踏面深度 `w=0.35-0.10d` m，范围 **0.25–0.35 m** | 难度越高，阶高越大、踏面越浅；平台宽 2 m，覆盖上/下楼 |
| boxes | 最高箱顶高度 `h=0.05+0.35d` m，范围 **0.05–0.40 m** | 对齐 PGMT 连续网格样式：箱宽 0.70 m、无间隙，每块 34×34 个箱格；箱顶采样于 `[0.05,h]`，中心 2 m 平台高 h，无负高度坑 |
| random rough | 输入噪声半幅 `A=0.01+0.07d` m，范围 **1–8 cm** | 恢复 PGMT 原生随机高度场：噪声步长 0.01 m、粗采样 0.30 m、网格 0.15 m、垂直量化 0.005 m、边框配置 0.25 m、陡面阈值 0.75；保留三次插值过冲，不把输入幅值当作最终高度硬上限 |
| up_stairs | 与 stairs 共用阶高、踏面深度公式 | 原生 `MeshInvertedPyramidStairsTerrainCfg`：中心低、四周高，从中央平台向外上楼；平台、边框与 stairs 相同 |
| up_slope | 与 slopes 共用坡角公式 | 原生 `HfInvertedPyramidSlopedTerrainCfg`：中心低、四周高，从中央平台向外上坡；平台、边框、分辨率与 slopes 相同 |

采用已安装 Isaac Lab 的原生分级方式：**第 L 行在生成时采样 `d=(L+u)/10`，`u∈[0,1)`**。固定 seed 后网格固定；reset 升降级改变所在 tile，不重新生成网格。范围端点是设计边界，L9 随机 tile 不保证恰好达到最大尺寸；`d=1` 另作上限验证。固定评测保存每块网格的实际 d 和几何，复用同一清单。

| 等级 | d 范围（右端不含） | 坡角（度） | 阶高（cm） | 踏面深度（cm，由易到难） | 箱高上限 h（cm） | rough 输入噪声半幅 A（cm） |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.0–0.1 | 0–3 | 5.0–7.5 | 35.0→34.0 | 5.0–8.5 | 1.0–1.7 |
| L1 | 0.1–0.2 | 3–6 | 7.5–10.0 | 34.0→33.0 | 8.5–12.0 | 1.7–2.4 |
| L2 | 0.2–0.3 | 6–9 | 10.0–12.5 | 33.0→32.0 | 12.0–15.5 | 2.4–3.1 |
| L3 | 0.3–0.4 | 9–12 | 12.5–15.0 | 32.0→31.0 | 15.5–19.0 | 3.1–3.8 |
| L4 | 0.4–0.5 | 12–15 | 15.0–17.5 | 31.0→30.0 | 19.0–22.5 | 3.8–4.5 |
| L5 | 0.5–0.6 | 15–18 | 17.5–20.0 | 30.0→29.0 | 22.5–26.0 | 4.5–5.2 |
| L6 | 0.6–0.7 | 18–21 | 20.0–22.5 | 29.0→28.0 | 26.0–29.5 | 5.2–5.9 |
| L7 | 0.7–0.8 | 21–24 | 22.5–25.0 | 28.0→27.0 | 29.5–33.0 | 5.9–6.6 |
| L8 | 0.8–0.9 | 24–27 | 25.0–27.5 | 27.0→26.0 | 33.0–36.5 | 6.6–7.3 |
| L9 | 0.9–1.0 | 27–30 | 27.5–30.0 | 26.0→25.0 | 36.5–40.0 | 7.3–8.0 |

v5 历史实现核对项（不覆盖当前 v6 选择）：① 原生 `HfRandomUniformTerrainCfg` 忽略 difficulty，包装幅值；② v5 使用上述正高度箱体，v6 已按用户要求改回原生 `[-H,+H]`；③ v5 包装踏深变化，v6 固定 0.32 m；④ 斜坡按角度线性映射；⑤ 检查网格实际高度/坡度及相邻落差。P3 容差规划改为当前 torso 所在 tile 等级的 `L/9`，出生课程仍独立记账；失败延迟单独固定，地图越界立即截断。

论文的 **37 cm 箱体**属于实机展示；在用户指定的 40 cm 新上限内，可单列为指定高度测试，但不再标记为训练范围外测试。上述参数已通过固定 v5 资产接入 P1 无感知训练；后续若调整范围，应升级地形版本并重新固定评测资产。

### P2：构建论文的 Terrain-Glimpse 与 IFM

**状态：P2.1–P2.4 已完成，P2.5 的 160 环境集成验收通过。** 指定 Stage1 迁移、GUI 25 次更新、新进程两次恢复、160 三组成本和 321 项 CPU/CUDA 回归均通过。1024 扩容受 PhysX 初始化显存限制，未通过，失效数据已剔除并加入错误终止保护。详见 [P2 计划](PGMT_P2_PLAN.zh-CN.md) 与 [P2.4/P2.5 验收报告](PGMT_P245_ACCEPTANCE.zh-CN.md)。

- [x] `PGMTPlacementEnvCfg` / wrapper 已提供 torso 中心、yaw 对齐的 **2×2 m、21×21** `terrain_map` 与 `terrain_valid`；v6 完整地图、负高度、边缘掩码和 reset 刷新已验收。P1 盲走配置仍移除扫描器，见 [扫描接口](PGMT_HEIGHT_SCAN.zh-CN.md)。
- [x] **P2.1 输入契约**：独立 `PGMTPerceptionEnvCfg`；2026-10-12 按用户要求升级 version=3：sensor−ground，高度无效值按 PMT 先填世界 Z=30 m，10% 丢点、±0.06 m 均匀噪声、观测裁剪 ±3 m；每次观测函数重新采样，单次生成地图与 mask。取消增强缓存和观测归一化，网络输入层 `PGMTTerrainInput` 负责裁剪后除以 3；Stage-1 策略仍盲。保留 2×2 m、21×21 和地图边界 mask，结果见 [P2.1 验收](PGMT_P21_ACCEPTANCE.zh-CN.md)。
- [x] **P2.2 Terrain-Glimpse（第 1–5 项完成）**：已实现输入、`1498→256→128→8` selector（中心范围 ±0.8 m）、掩膜加权双线性 5×5 裁剪，以及共享 `52→128→128` patch MLP。点有效权重至少 0.5，patch 至少 13/25 点有效；完整入口返回 4 个 128 维 token、bool token 掩膜及可选独立调试副本。157 项相关回归一次完整运行通过、无跳过；20 张真实地图及缺测压力样例通过 CPU/CUDA 和独立 NumPy 对照，token 损失经裁剪高度回传 selector。5 个种子的中心分布及 batch 1/160/1024/8192 耗时/显存已记录；独立验收通过。详见 [P2.2 验收记录](PGMT_P22_ACCEPTANCE.zh-CN.md)。
- [x] **P2.3 IFM（第 1–5 部分完成）**：13 个 K/V、逐环境掩膜、9-key 回退、共享 actor/critic、观测分组、version=3 元数据及整网诊断完成。原生 PPO 保存/重算/梯度通过；20 张真实地图扩展为 100 个样例，CPU/CUDA 的动作/value/intent 最大误差 <1.5e-7，disabled 与旧行为逐值一致。285 项相关回归无跳过，整次融合验收执行 624 次 Adam 更新；完整 actor/critic 与 PPO 的 batch 160/1024 成本已记录，GPU 并行负载限制明确注明。详见 [P2.3 融合验收](PGMT_P23_ACCEPTANCE.zh-CN.md) 和 [P2 计划第 3.7–3.12 节](PGMT_P2_PLAN.zh-CN.md)。
- [x] **P2.4 迁移与训练入口**：独立配置/入口及 PGMT runner 完成；指定 Stage1 的 49 项状态完整迁移、10 项新感知初始化，新 Adam/iteration=0/全新 sampler 与课程。P2 严格恢复全部持久状态、学习率、下一轮编号及 RNG；实际 25→30→再次加载 30 逐项通过。
- [x] **P2.5 小规模验收**：实际禁用感知与 Stage1 输出误差 0；五类、L0/L5/L9 几何/边界、GUI 25 次 PPO、参考/接触/sampler 结算和二次恢复通过。160 环境盲走/仅扫描/完整感知成本已记录；321 项 CPU/CUDA 回归通过。**1024 扩容未通过（PhysX 显存容量），不计为完成的大规模训练能力。**

**阶段边界：P2 短训保留三头与原奖励，第四头和 terrain-contact 奖励在 P3 同时接入。** 正常感知模式默认联合训练；仅在短训发现注入不稳定时采用可选 attention bias 退火，最终归零。P2 完成代表感知可训练、可续训，不代表地形适应性能或完整 Stage 2 已完成。

### P3：闭合奖励、地形课程与 PPO（recovery 暂缓）

2026-10-13 已完成进入 P3 前的源码/论文审计，详见 [P3 奖励审计与实施建议](PGMT_P3_REWARD_AUDIT.zh-CN.md)。2026-10-14 新正式任务 `pgmt_stage2_terrain` 已启用 **27 项奖励、四头及三项下肢容差**，原 `pgmt_stage2` 保持 21 项/三头。容差使用当前 torso 所在 tile 的 family/level，与出生课程记账分开。首轮旧配置的 415 项回归和 3+2+1 次更新保留于 [P3.4 验收](PGMT_P34_FOUR_HEAD_ACCEPTANCE.zh-CN.md)；最新配置的 600 项回归、160 环境 3+2 次更新及二次恢复见 [P3 总验收](PGMT_P3_ACCEPTANCE.zh-CN.md)。实际 USD 没有独立 head 刚体，已确认 torso 四个 capsule 的静态归属，动态冲击与必要接触惩罚统计仍待完成。

- [x] **接触状态与事件基础**：Stage2 每个控制步维护左右脚的原始接触、地形向上力承重、状态年龄、连续时长、开始/结束/切换及 touchdown 事件。地形力过滤覆盖所有有效 tile；缺测不视为摆动。reset 首次测量只建基线，终止步事件与参考标签冻结保存，同一步重复读取不推进计时。70 项 CPU/CUDA 回归及 5 环境 Isaac 验证通过，接口、结果和运行命令见 [接触状态说明](PGMT_CONTACT_STATE.zh-CN.md)。六项 reward 数值核、四头和正式 PPO 接入均已完成，见下表及 P3.4 验收。

- [x] **T1 touchdown quality 数值核与运行期接入**：3×3 点、20×10 cm、中心 +3.9 cm、每脚 yaw；有效地图和 2×2 m 扫描覆盖内的 9 个干净高度计算总体方差，质量为 `exp(-variance/0.03²)`。2026-10-18 改为双脚实际承重与有效参考标签整体匹配时持续结算，按参考支撑脚数取平均；无支撑、失配或任一标签/力测量无效时为 0。保留原 T1 接口与 +10 权重，总上限为 0.2/步，真实 touchdown 仅保留作诊断。正式四头任务 terrain 组已接入，当前与历史验收分列于 [Touchdown quality](PGMT_TOUCHDOWN_QUALITY.zh-CN.md)。

- [x] **T2 reference contact match 数值核与运行期接入**：按用户确认使用全对象法向接触力范数 `>10 N`，与同帧离线标签逐脚匹配，按有效参考数量平均；实际缺测不获匹配分，参考 unknown 不作 swing。持续奖励，独立权重 +1.5，当前全匹配每步 0.03。107 项 CPU/CUDA 回归、5 环境/72 步 Isaac 与 NumPy 对照通过；T1+T2 的独立 RewardManager 共 23 项，新四头任务已正式启用。见 [T2 实现与验收](PGMT_REFERENCE_CONTACT_MATCH.zh-CN.md)。

- [x] **T3 slip 数值核与运行期接入**：全对象接触 c 门控实际双脚的世界 XY link 原点速度平方和，持续计罚，独立权重 −1.0。首次测量、参考 unknown、重复读取及终止/局部 reset 语义验证通过。125 项 CPU/CUDA 回归、5 环境/72 步 Isaac 与独立 NumPy/质心速度重建对照通过；T1–T3 独立 RewardManager 共 24 项，新四头任务已正式启用。见 [T3 实现与验收](PGMT_SLIP.zh-CN.md)。

- [x] **T4 stumble 数值核与运行期接入**：按全对象脚部力严格判断 `||Fxy|| > 5|Fz|+10 N`，逐脚求和、持续计罚，独立权重 −20；不以参考标签或地形承重门控。164 项 CPU/CUDA 回归、5 环境/102 步 Isaac 通过；增加真实侧撞场景补足正例，2 次触发累计约 −0.8，NumPy 代价率误差为 0。T1–T4 独立 RewardManager 共 25 项，新四头任务已正式启用。见 [T4 实现与验收](PGMT_STUMBLE.zh-CN.md)。

- [x] **T5 contact switching 数值核与运行期接入**：复用全对象接触 c 的双向快速切换事件，旧状态年龄 `<0.1 s` 且本回合步数 `>5` 时逐脚计罚，独立权重 −30；保持状态不重复计罚，首次测量、缺测恢复与 reset 不产生伪事件。195 项 CPU/CUDA 回归、5 环境/102 步 Isaac 通过；独立原始力历史重放核对 4 次触发（进入/退出各 2 次），累计约 −2.4。T1–T5 独立 RewardManager 共 26 项，新四头任务已正式启用。见 [T5 实现与验收](PGMT_CONTACT_SWITCHING.zh-CN.md)。

- [x] **T6 contact force 数值核与运行期接入**：对左右脚全对象 XYZ 力平方和持续计罚，无力阈值/接触门控，独立权重 −1e−6。223 项 CPU/CUDA 回归、5 环境/102 步 Isaac 通过；552 个非零力样本含 24 个不超过 10 N 的样本，NumPy 最大相对误差约 1.15×10⁻⁷。T1–T6 全部完成并通过 27 项/四组独立 RewardManager 联验；新四头任务已正式启用。见 [T6 实现与验收](PGMT_CONTACT_FORCE.zh-CN.md)。

- [x] **三项下肢跟踪容差数值核、接口与独立验收**：按当前 torso 所在 tile 的 family/level，使用 `max(0,e−α×χ×m×clip(L/9,0,1))`，error 保持非负。三个基础上限、地形白名单、等级缩放和逐部位 α 均可配置，默认参数沿用规划。267 项 CPU/CUDA 联合回归、补充后 30 项容差专项、5 环境/14 步 Isaac 覆盖 50 区域通过，含跨 tile、终止/reset 与越界。原 P2 默认关闭，新四头任务默认开启；接口、记录与续训契约见 [容差实施计划与验收](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)。

| 必须实现的机制 | 接入要求 |
| --- | --- |
| 跟踪容差 | **已实现并验收，新四头任务默认开启。** 逐 link/joint 原始误差计算 `max(0,e−α×χ×τ)` 后进入指数核。默认仅 stairs/slope/boxes 的**下肢位置、姿态和关节位置**放宽；适用地形和上限可配置。family 与 L 取当前 torso 所在 tile，默认 `τ=m×clip(L/9,0,1)`；出生等级和 manifest 连续 difficulty 不参与计算。P2 默认关闭，详见[配置与验收](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)。 |
| Terrain-contact 奖励 | **六项数值核、缓存、独立联验和正式四头 PPO 接入均已完成。** 表 I 权重：touchdown quality **+10**、reference contact match **+1.5**、slip **−1**、stumble **−20**、contact switching **−30**、contact force **−1e−6**。落脚质量使用实际落脚处干净高度方差；区分事件与持续项，reset 清空当前历史，旧回合结算快照保留；无效/观测外采样不能算优质落脚点。 |
| 原有奖励 | 保留既有权重和全局位置反馈：修正速度同时用于 command/reward，静止参考修正为零。按表 I 审计 impact 身体集合，当前仅取 torso。 |
| 四头 PPO | **已完成。** critic 和 `extras["reward_groups"]` 均为 `[N,4]`，组顺序 `[upper,lower,terrain,auxiliary]`；四组和等于标量 reward，dt 一次。独立 GAE/return、优势求和后归一化、四头 value loss 与梯度、timeout/done 数值验收通过，新增各头 value loss 日志。 |
| 地形课程 | **已实现核心课程及续训恢复**：完整动作跟踪升一级，跟踪失败降一级，L0–L9 封顶；跨 tile 不改写出生分配，地图越界/提前超时保持等级。六类结果按出生 family/level 记账，reset 更新 origin，各回合只结算一次。奖励容差另取当前位置 tile 等级；前四项跟踪失败按用户指定阈值连续超限 0.2 s 才终止，各项/各环境独立计时，恢复正常或 reset 清零。验证见 [地形课程](PGMT_TERRAIN_CURRICULUM.zh-CN.md)。 |
| 终止条件 | **按用户指定已实现**：pelvis XY 轨迹误差 >1 m；torso Z 与当前实际 torso XY 的地形高度＋原动作 torso Z 比较，误差 >0.4 m；只检查双手腕相对 pelvis 的高度误差 >0.4 m。保留姿态阈值 0.8；超时改为 10 s/500 步；动作耗尽、地图越界显式截断保留。前四项连续超限 0.2 s（10 控制步）后终止，无额外 reset 宽限，见 [终止说明](PGMT_STAGE2_TERMINATIONS.zh-CN.md)。 |
| 保存/续训 | **四头已完成并原生验收。** Stage1/P2 三头通过显式迁移保留旧输出行，新 terrain 行为零，Adam/迭代/课程/sampler 新建。同配置四头严格恢复模型、归一化、Adam、LR、iteration、sampler/概率、地形课程/统计和 RNG；校验奖励/容差/感知契约。物理回合重新初始化，不伪造中断 outcome。 |

**暂缓事项：** recovery pool、倒地初始化、survival-based recovery curriculum、恢复比例日程、恢复专属终止宽限及 **+12.5 recovery upward velocity**。保留 `model_15000.pt` 作为 stage 2 初始化来源，训练无需先补恢复。本轮目标奖励数为 **27 项：上肢 6 + 下肢 6 + terrain-contact 6 + auxiliary 9**；新正式任务已启用 27 项，原 P2 三头任务保留 21 项。四头顺序为 `[upper, lower, terrain, auxiliary]`。本轮成果按“未包含 recovery 的 PGMT 变体”记录。

**Stage2 辅助奖励调整：** `auxiliary_end_effector_acceleration` 仅计算双腕 XYZ 加速度与参考加速度的均方差（分母 6），不包含双脚；权重 −0.001、均值后截断上限 10000、dt 与 reset 差分历史逻辑沿用原实现。Stage1 仍计算双脚/双腕。配置接口为 `pgmt_ee_acceleration_body_names`，随环境配置进入严格续训契约，详见 [奖励公式 A6](PGMT_P3_REWARD_AUDIT.zh-CN.md)。

### P4：分级验证、正式训练和固定评测

**2026-10-14 已开始执行数值/短训与正式训练准备。** 623 项 CPU/CUDA 数值回归通过；PhysX 共享地形粗筛修复已接入正式入口并通过环境隔离/跨 tile 接触验证；平地 160 环境 3 更新、五类地形 160 环境 25+5 更新及再次独立恢复通过，1024 环境已完成 3 次 PPO 更新。正式训练指定 `cuda:0`，从 `model_15000.pt` 迁移，优先 8192 环境、OOM 降档；完整过程、日志指标、时间预算及实时运行入口见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。固定评测与部署导出尚未因此完成。

**正式运行停止记录（2026-10-15）：** 1024 的新进程恢复和到时保存验证完成（累计 4 更新）；`cuda:0`、8192 环境正式 run 完成 **383 次更新、75,300,864 transitions**，无 OOM、未降环境数。用户停止请求在更新边界生效，`final.pt` 已保存，监督状态为 `completed / stop_file`；未跑满原定 12 小时或 20,000 次更新。训练记录不等于收敛或固定评测通过。后续采样耗时诊断见 [性能记录](PGMT_COLLECTION_PERFORMANCE.zh-CN.md)。

1. **数值验证：** 权重映射、地图/梯度、容差范围、接触对齐、四头 GAE/timeout、缓存重置，以及 stage 1 回归。
2. **Isaac 短训：** 当前 v6 使用 160 环境（每类 32）、20–50 次更新，覆盖五类地形、地形课程、换片和续训；再用 1,024 环境测吞吐/显存，按资源扩至 8,192。检查 PhysX 碰撞容量；沿用 P1 已验证的 256 MiB 碰撞栈起点。旧 v5 七类仅作独立回退回归。
3. **正式训练：** 从指定 checkpoint 开始新 run；先完成平地迁移回归，再开放低级地形和感知，最后运行地形课程。保留平地样本防止遗忘，日志记录各 family/level 成功率、上/下肢误差、四组回报和 value loss、KL、terrain attention 及课程分布。
4. **固定评测：** 单独建立 30 s / 1,500 控制步的连续回合，当前 v6 五类十级各 192 回合，共 **9,600** 回合，L9 共 **960**；stairs/slope 已采用倒金字塔与随机 XY/yaw，不重复计为额外样式。固定 motion/start frame、terrain、小块/出生位姿、随机化、观测噪声与执行延迟，并复用同一清单比较 stage 1 与 stage 2；评测禁用自适应采样和地形升级。先审计当前 clipped 片段是否有足量 30 s 窗口，不足则另建并披露合格评测数据；禁止中途重置机器人、把 chunk 结束或训练期 10 s/历史 15 s timeout 算作成功。v5 如另行评测，单独报告。
5. **结果与导出：** 报告整体/L9/各地形完成率、body position/orientation、joint RMSE、contact F1 和平地遗忘，注明未使用 recovery pool/奖励。用地图扰动诊断感知依赖；论文消融需另行重训。扩展现有 MuJoCo contract/export，验证地图输入、同状态动作和地形闭环。

论文完成率 87.81%/L9 83.33% 仅作参照，本地 benchmark 不保证与作者等价。总预算按吞吐确定，图 4 的 10k 不是训练总步数。真实 LiDAR/遥操作接入放在后续部署。

## 3. 首版配置与代码落点

以下是**工程起点，不是论文公开超参数**；集中写入 stage 2 YAML，评测前冻结。

| 配置 | 建议起点 |
| --- | --- |
| 地形 | 当前 v6：五类十级、初始 L0–L1；每区域 16 个 6×6 m 小块；坡角 0°–30°、平台 0.5 m；阶高 0.05–0.30 m、踏深固定 0.32 m；boxes 原生正负幅度 H=0.05–0.40 m、无中央平台、连接总宽 0.4 m；rough 输入噪声半幅 1–8 cm。旧 v5 保留回退。 |
| 感知 | selector MLP 256/128，共享 patch MLP 52→128→128、ELU 隐藏层，128 维 token；PMT sensor−ground 符号、世界 Z=30 m 无效填充、10% 丢点、±0.06 m 均匀噪声、±3 m 观测裁剪；网络输入层除以 3。增强每次观测调用重采样；4 个 terrain token/掩膜已接入 IFM，正式 `pgmt_stage2` 配置默认开启感知。 |
| 容差 | 下肢 position/orientation/joint position 上限先用 0.20 m / 0.7 rad / 0.5 rad；受影响元素 `alpha=1`，其余为 0，保留逐元素覆写能力。 |
| 优化 | P2 实际配置继承 rollout=24、epochs=5、minibatches=4、gamma=0.99、lambda=0.95 和现有优势处理；初始 LR=1e−4，adaptive 调整，全部参数联合训练，无旧主干单独学习率系数。探索标准差继承 checkpoint；P3 的进一步调参需另行验证。 |
| 未定参数 | 训练总预算及后续性能调参待确定；前四项跟踪终止已按用户指定增加 0.2 s 连续容忍时间，reset 清零，无等级相关延迟或额外出生宽限。接触奖励公式/阈值已接入四头任务；直接开启感知，无注入退火日程。恢复比例日程暂缓。 |

| 改动范围 | 建议文件 |
| --- | --- |
| 任务与配置 | 新增 `pmt_tasks/env_cfgs/pgmt/stage2.py`、各轴 `pgmt_stage2.yaml`；扩展 `agent_cfgs/pgmt.py`、`builder.py`、`registry_gym.py` |
| 地形/数据/reset | 新增 PGMT terrain、command、contact preprocessing 模块；复用 `utils/terrain.py`，扩展 `pgmt_runtime.py` 的参考变换与生命周期 |
| 网络/初始化 | 扩展 `networks/pgmt_actor_critic.py` 的 stage 2 分支、显式权重迁移及 checkpoint metadata；保留 stage 1 默认行为 |
| 奖励/课程/终止 | 扩展 `pgmt_math.py`、`mdp/pgmt_observations.py`、`mdp/pgmt_rewards.py` 及 PGMT curriculum/termination 模块；recovery 模块暂缓 |
| 训练/评测 | 仅在必要处扩展 runner 的初始化/保存；新增 `validate_pgmt_stage2.py`、固定 benchmark 清单及评测入口，复用现有 MuJoCo 导出框架 |

**本轮完成标准：** 指定权重可审计迁移；当前 v6 五类十级、随机出生、四 glimpse、四头价值、地形容差、六项接触奖励及地形课程形成闭环；训练/续训与固定 30 s 评测通过，输出完整配置、假设清单和结果。recovery 不计入本轮验收条件。旧 v5 七类资产与入口独立保留。本轮机制完成、能够训练与达到论文性能分别验收。

## 依据与复用入口

- [论文](../PGMT_LITERATURE.pdf)：p.3 观测与 IFM；p.4 表 I、恢复课程与 glimpse；p.5 地形容差、课程、四头和评测；p.6 结果与消融。
- [现有 stage 1 说明](PGMT_STAGE1.zh-CN.md)、[网络](../motion_tracking_rl/networks/pgmt_actor_critic.py)、[环境](../pmt_tasks/env_cfgs/pgmt/stage1.py)、[参考/缓存](../pmt_tasks/pgmt_runtime.py)、[PPO storage](../motion_tracking_rl/storage/rollout_storage.py)。
- 本地独立实现：[glimpse](../../PGMT/pgmt/modules.py)、[地形](../../PGMT/pgmt/terrain.py)、[兼容采样](../../PGMT/pgmt/terrain_training.py)、[工程配置](../../PGMT/configs/optimized_v2.yaml)、[碰撞容量问题记录](../../PGMT/reports/STAGE2_STARTUP_FIX.zh-CN.md)。仅迁移已核验组件，统一使用 PMT 的训练和控制契约。
