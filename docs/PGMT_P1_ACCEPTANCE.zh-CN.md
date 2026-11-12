# PGMT P1 验收与无感知地形训练记录

日期：2026-10-10。验收目标是确认 P1 的地形、初始化和回合生命周期，并使用已有 Stage 1 权重启动真实 Isaac PPO 训练。此次保持 Stage 1 网络与三头奖励结构，不加入 Terrain-Glimpse/IFM，也不把高度图送入 actor/critic。

## 当前验收结果

P1 五组独立检查已通过。修复地形内嵌 PhysicsScene 后的最终汇总报告为 [`logs/pgmt_p1_acceptance/single_scene/acceptance.json`](../logs/pgmt_p1_acceptance/single_scene/acceptance.json)。GUI 训练结果见下文；检查既包括实际 PPO 参数更新，也包括视口与物理姿态同步。

| 检查 | 覆盖范围与结果 | 报告 |
| --- | --- | --- |
| 整库第 0 帧 | 862 个动作；每级 5,316 个 CSV 允许组合；七类十级、三个 yaw，共 159,480 个候选放置。无关节越界；两个候选被几何检查拒绝后均通过真实 command 重试安全初始化。接受姿态最小保守碰撞间隙约 2.990 mm | `start_poses/validation.json` |
| 真实初始化与碰撞 | 兼容全部地形的 walk 动作，70 块地形 × 三个 yaw，共 210 次 reset；650 个 PhysX/GPU/资产探针，高度最大误差约 4.04 μm；无地形重叠，当前/未来/command/reset 的统一变换一致 | `physical_placement/validation.json` |
| 按位置查询与跨 tile | 全部 70 块地形、720 个探针；跨等级、跨样式、双脚跨缝查询不修改出生分配。参考或机器人超出有效地图立即 truncation | `map_queries/validation.json` |
| 接触与采样 | 全库接触标签与原 NPZ 对照；当前/未来/局部环境接口及片段末尾无效 mask；旧奖励读旧标签、reset 后读新标签；动作级自适应与 CSV 排除生效 | `contact_sampling/validation.json` |
| 回合、课程、续训 | 998 个真实控制步；旧奖励先结算，sampler 单次记账，完整跟踪升一级、失败降一级、中性截断保持；实际 runner 保存恢复课程与 sampler，恢复后继续运行 | `curriculum_lifecycle/validation.json` |

表中报告路径均位于 `logs/pgmt_p1_acceptance/single_scene/`。86 项相关单元/回归测试通过。课程审计使用受控终止标志验证状态转换；它不代表策略跟踪成功率。整库检查覆盖初始化几何，独立 PhysX 重叠检查使用代表动作，未宣称逐帧验证所有动作在所有地形上均可完成。随机起始帧保持关闭，未把随机帧初始化列为已验收。

## 训练逻辑

1. **初始化权重**：唯一输入为 `logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt`，iteration=15000，SHA256=`40708024acb51c6ce3cd87d6a067f59ff822763b6d8b73d3e2473d9953793087`。新日志目录独立于源训练目录，不覆盖该权重。
2. **无感知配置**：`PGMTBlindTerrainEnvCfg` 继承已完成的 PGMT 地形 command、参考放置及课程，关闭扫描器，观测只有 `proprio[96]`、`proprio_history[10,96]`、`command_window[8,61]`、`critic[251]`。网络为原 `PGMTActorCritic`，embedding=128、四个 attention head、actor 输入 224、critic 输入 379、29 个动作、upper/lower/auxiliary 三个 value head。
3. **严格载入**：核对 checkpoint 的网络签名和完整观测 schema；逐张量核对载入值，恢复归一化、优化器与 iteration。Stage 1 checkpoint 没有地形课程状态，因此地形等级与课程/sampler 统计从新状态开始。优化器恢复后的实际学习率同步到 PPO 的自适应学习率变量，避免从 YAML 默认的 1e-3 意外重启；该 checkpoint 的学习率约 1.31687e-4。
4. **动作与出生**：读取 `pgmt_clipped_50hz_tcrs_v1` 全库及原兼容表。`sampler_type=adaptive`、`enable_frame_sampling=False`、`chunk_length=0`、`fixed_start_frame=0`、`require_valid_support_start=False`。先依据出生地形过滤不兼容动作，再按动作级失败统计采样；保留膝部支撑爬行动作。
5. **地形放置**：在原生 origin 附近采样 XY/yaw，以动作第 0 帧建立固定刚体变换。全身保守碰撞覆盖与实际网格高度/距离决定 Z 位移；允许原生负高度，原始关节轨迹不改写。拒绝不安全候选后重试；不能把被拒绝的候选当作已初始化的机器人。
6. **控制与优化**：原生 50 Hz 控制、每环境 24 步 rollout、原 PPO 配置；实际训练保留原噪声、事件、跟踪终止与课程。每步先计算旧参考下的 termination/reward，再结算课程和 sampler，随后 reset、推进新 command、计算观测。动作耗尽显式 done，切断旧 GAE；不在无 done 的轨迹中直接换片。
7. **跨 tile 与 reset**：参考刚体变换在回合内保持不变；地形查询随当前世界位置更新，课程等级沿用出生分配。接触标签跟随当前 motion/frame，不使用旧回合缓存。reset 后从第 0 帧重置机器人，原生自动 reset 后的返回观测可能已推进到第 1 帧。
8. **训练审计**：运行中检查有限观测、奖励与损失、参数实际更新、奖励分组求和、非 done 环境的参考变换不漂移、新接触标签索引、初始化位置一致性与 sampler 不重复结算。输出 TensorBoard、逐更新 `progress.jsonl`、`training_status.json`、验收首轮 checkpoint 和常规 checkpoint。

## 发现的问题与处理

| 环节 | 问题与证据 | 处理方式 |
| --- | --- | --- |
| 验收覆盖范围 | 原放置验收主要使用代表动作，不能证明整库初始化安全 | 新增 `validate_pgmt_start_poses.py`，遍历全部 CSV 允许组合；保留代表动作的独立 PhysX 碰撞检查 |
| 候选放置 | `jumps1_subject2_clip20` 与 `walk3_subject1_clip21` 在 up_stairs/L5、yaw=−135°、中心 XY 的首次候选不满足碰撞/支撑约束 | 现有几何检查正确拒绝；验证原生 command 重新采样 XY/yaw 后均安全接受，间隙约 3 mm。未删除动作，未改 CSV，未改关节轨迹；详细失败与重试参数已留档 |
| 训练入口 | 既有 `train.py --task ...` 注册的是 Stage 1 平地任务，不能直接代表已完成的 P1 地形逻辑 | 新增明确的 `train_pgmt_p1.py` 与 `PGMTBlindTerrainEnvCfg`，使用原生 OnPolicyRunner/PPO，保留严格相同的 checkpoint 输入结构 |
| 学习率恢复 | 原 runner 恢复 optimizer.param_groups 的学习率，但 PPO 独立的 `learning_rate` 标量仍可能保留配置初值 | 本验收入口载入后显式同步，记录实际初始与逐轮学习率 |
| Isaac 5 GUI | 两张 GPU 上的可见窗口探测均在 RTX/native 插件处崩溃；关闭 SyntheticData 菜单/Fabric 渲染路径也未解决 | 保留失败日志，改查本机 Isaac Sim 6；没有把 headless 训练或外部回放窗口冒充真实 GUI 训练 |
| Isaac Sim 6 接口 | 旧 Isaac Lab 引用 `omni.physics.tensors.impl.api`、SoftBody 类型注解和旧 deformable schema，Sim 6 已改名/迁移 | 增加仅此入口显式启用的 `isaacsim6_legacy.py` 兼容层；不修改安装环境，不替换刚体物理、策略或奖励公式；兼容范围限定为本次 G1 刚体场景 |
| 启动顺序 | 在 AppLauncher 前通过 `pmt_tasks.utils` 导入兼容层，会被该包的 eager import 提前加载仿真模块，导致 carb settings 尚未初始化 | 将启动兼容层移至纯 Python 顶层模块，导入桥在 Kit 启动前安装，schema 适配在 Kit 启动后执行 |
| Kit 与 Python 包组合 | 旧 Kit 配置固定旧 URDF 插件；混入 Sim 6 的新版 `isaaclab_contrib` 又引用旧 Lab 不具备的模块 | 使用 Sim 6 原生 `isaacsim.exp.base.python.kit`，而四个 Isaac Lab Python 包全部显式使用 PMT 原来的源码路径，避免混搭 |
| G1 资产转换 | 旧 Lab 转换器依赖的 URDF Python 接口在 Sim 6 已变更 | 复用此次原生 Isaac 5 物理验收生成的同一 G1 USD；保留执行器、刚体和接触参数。副本位于 `logs/pgmt_p1_acceptance/native_robot/`，`provenance.json` 记录 URDF/转换产物哈希与来源 |
| 地形物理时钟 | 地形预览 USD 自带 PhysicsScene，导入后与训练 SimulationContext 并存；出现 `Physics scenes stepping is not the same` | `PGMTTerrainImporter` 仅在训练 stage 停用地形下的内嵌 PhysicsScene，使用唯一主场景的 200 Hz 物理时钟；原地形资产不改写。修复后完整重跑五组 P1 检查通过，实际 GUI 运行另做 720 次 PhysX 探针检查 |
| TensorDict 版本 | 新版 TensorDict 的迭代语义不适合 `set(observations)` | 显式核对 `.keys()`，继续严格核对四组无感知观测与 checkpoint schema |
| GUI 姿态同步 | 真实 PPO 已更新，但早期截图只有背景/地形；GPU tensor 模式下 USD pelvis 保持初始配置，强制 USD 刷新也不能作为 GPU 姿态渲染路径 | 恢复 `use_fabric=True` 与 Sim 6 原生 Fabric/Hydra 默认设置，调用 `SimulationManager.enable_fabric(True)`，由 Lab 的原生 `forward()` 刷新 GPU 姿态。相机显式跟踪实际 viewport。r14 的第 1/5 次更新截图均能看到机器人，姿态随训练变化；最终代码已移除无效的强制 USD 刷新尝试 |

GUI 尝试日志依次为 `logs/pgmt_p1_gui_probe*.log`、`logs/pgmt_p1_train_gui*.log`；每次训练尝试使用独立目录，保留失败信息，不覆盖前次记录。r10 已完成 90 次 PPO 更新，r11 已完成 61 次；为修复画面而人工停止并从用户指定权重重新验收，未将这些尝试冒充最终可视化验收结果。

## 真实 GUI 训练证据

最终运行目录：[`logs/pgmt_p1_blind/acceptance_gui_20261010_r14`](../logs/pgmt_p1_blind/acceptance_gui_20261010_r14)。历史进程 PID=1426901，启动于 2026-10-10 19:23:02（Asia/Shanghai），140 个环境，GPU `cuda:1`，`headless=False`，原计划追加 1000 次 PPO 更新。状态文件记录已于 19:43:51 按用户要求停止，实际完成 390 次更新；本次验收要求启动成功及连续训练检查，未完成计划中的 1000 次更新。

- 49 个 checkpoint 张量精确载入，网络签名与观测 schema 均相同；实际只有 Stage 1 四组观测，没有扫描器与感知输入。
- Sim 6 实际训练场景的 720 个 PhysX 高度探针通过，最大高度误差 **2.176 μm**。
- 第一次真实 PPO 更新已保存 `acceptance_first_update.pt`；离线读取确认其包含 `environment_state_dict.curriculum` 与 `sampler`、优化器及策略元数据。
- [`training_viewport_1.png`](../logs/pgmt_p1_blind/acceptance_gui_20261010_r14/training_viewport_1.png) 与 [`training_viewport_5.png`](../logs/pgmt_p1_blind/acceptance_gui_20261010_r14/training_viewport_5.png) 已人工检查，机器人姿态不同，画面来自训练实例本身。窗口内 **PGMT P1 | Live blind PPO** 面板可点击七种地形按钮，跟随对应环境；隐藏其他机器人只改变可见性，所有环境仍在训练。
- 源 Stage 1 checkpoint 的 SHA256 复查保持不变。持续状态见 [`training_status.json`](../logs/pgmt_p1_blind/acceptance_gui_20261010_r14/training_status.json)，训练控制台见 [`pgmt_p1_train_gui_r14.log`](../logs/pgmt_p1_train_gui_r14.log)。

启动验收快照截至 **2026-10-10T19:25:53.972724**：已完成 **49 次 PPO 更新**、**1176 个控制步**、**406 次 reset / sampler outcome**；**164,234 次参考变换检查**和 **164,640 次接触索引检查**通过，损失、观测、奖励及参数均为有限值，策略参数已实际改变。完整跟踪成功 200 次、跟踪失败 206 次，课程已实际升降级。冻结证据见 [`acceptance_snapshot.json`](../logs/pgmt_p1_blind/acceptance_gui_20261010_r14/acceptance_snapshot.json)。

**版本整理时核对的停止状态：** `training_status.json` 为 `stopped_by_user`，最后更新于 19:43:05，停止记录为 19:43:51；共 **390 次 PPO 更新、9360 个控制步、3572 次 reset / sampler outcome**，参考变换/接触索引检查分别为 1,306,828 / 1,310,400 次。课程累计完整跟踪成功 1724 次、失败 1848 次。最后迭代为 15389，最后常规保存文件为 `model_15350.pt`，两者不等同。此结果与原始证据哈希一并冻结在 [版本记录](../CHANGELOG.md) 和 [版本快照](versions/pgmt-p1-2026.10.10.json)；本次整理未重启训练。

状态文件中的 `gui.usd_root` / `usd_pose_error_m` 是保留的 USD 路径诊断量。最终采用 Fabric 时，USD 不逐帧回写 GPU 姿态，因此该差值不表示实际渲染误差；物理 root、Fabric 动画和实际截图用于确认当前显示。训练数值正常仅说明闭环可训练，并不表示此时已在困难地形收敛。

## 复现入口

P1 独立检查：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/accept_pgmt_p1.py --device cuda:0
```

当前机器的 GUI 入口：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash scripts/train_pgmt_p1_gui.sh \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --num_envs 140 --max_iterations 1000
```

该封装使用本机 `isaacsim` 环境的 Python 3.12 / Isaac Sim 6.0.1，配合 PMT 原来的 Isaac Lab Python 源码、审计过的 G1 USD 和显式兼容开关，默认 `cuda:1`、`headless=False`。不要同时启动多个相同规模的验收任务。P1 独立检查继续使用 `pmt` 环境的 Python 3.11 / Isaac Sim 5.1。

### 视口模式切换

**PGMT P1 | Live blind PPO** 面板提供三种模式：

- **Global / free camera**：显示全部机器人，一次性把相机移到全地图俯视位置，随后解除自动跟随，可自由旋转、平移和缩放；再次点击可回到全景。
- **Follow selected robot**：恢复跟随上次选中的机器人，并隐藏其他机器人。七个地形按钮会选择对应地形的第一个环境并进入单机器人跟随。
- **View tile**：在下拉框选择七种地形中的一种及 **L0–L9**，点击按钮将相机定位到该固定地块。保留所有机器人可见，可以观察实际经过此地块的机器人；即使当前没有机器人，也可查看地形。机器人 reset、跨 tile 或课程升降级均不会移动该视角。

默认仍为跟随模式。启动命令末尾加 `--viewer_mode global` 可直接进入全局自由视角。切换只修改 USD 可见性与相机；所有环境持续训练，策略、奖励、采样、课程和 reset 逻辑保持一致。全地图范围较大，全景中的机器人较小，可手动靠近目标区域观察。上述按钮是原始 P1 验收后的显示功能补充，不改写已冻结的验收记录。

tile 模式中的 **Camera offset (m)** 可设置相机相对该地块原生 origin 的世界坐标偏移 `dX/dY/dZ`，点击 **View tile** 应用。相机朝向地块 origin，保留倒金字塔的负高度；随后可手动移动相机，再次点击按钮可回到设定位置。面板显示地块名、资产实际 difficulty 和 origin 坐标。这里选择的等级只用于查看，不修改训练环境的课程等级或出生分配。

例如，启动时查看 `up_stairs` 的 L5，并指定相机偏移：

```bash
bash scripts/train_pgmt_p1_gui.sh --num_envs 140 --max_iterations 1000 \
  --viewer_mode tile --viewer_terrain up_stairs --viewer_level 5 \
  --viewer_tile_offset 20 -28 48
```

`scripts/train_pgmt_p1.py` 的 `--max_iterations` 表示载入后追加的 PPO 更新次数，默认 1000；日志目录默认按时间生成，也可显式指定新的 `--log_dir`。每 50 次更新保存常规 checkpoint，第一轮另存 `acceptance_first_update.pt`。继承 runner 的文件编号习惯，第一次更新之后的常规文件可能仍命名 `model_15000.pt`，但位于全新验收目录，源权重不会覆盖。

本次是 P1 无感知训练启动验收：七种样式均参与，出生等级从 L0–L1 开始按课程变化。五类语义组各 20% 的分配仍待后续接入；当前保留原生七列布局分配。Terrain-Glimpse/IFM、第四个价值头、地形接触奖励、恢复机制及论文最终性能不属于本次通过范围。
