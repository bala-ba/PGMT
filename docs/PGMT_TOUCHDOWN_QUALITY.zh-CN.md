# PGMT P3：Touchdown quality

**2026-10-18：T1 已改为“双脚整体支撑匹配时持续结算”，保留 `terrain_touchdown_quality` 与 `touchdown_state()` 接口。配置版本为 `support_pattern_mean_v2`。下方当前公式覆盖历史事件奖励；2026-10-14/15 验收及旧训练曲线仍对应旧规则。**

日期：2026-10-14；状态更新：2026-10-15。T1 数值核、真实地形查询、控制步缓存及独立 RewardManager 配置已实现并验收。同日 [T2 reference contact match](PGMT_REFERENCE_CONTACT_MATCH.zh-CN.md)、[T3 slip](PGMT_SLIP.zh-CN.md)、[T4 stumble](PGMT_STUMBLE.zh-CN.md)、[T5 contact switching](PGMT_CONTACT_SWITCHING.zh-CN.md)、[T6 contact force](PGMT_CONTACT_FORCE.zh-CN.md) 均已完成，并通过六项联合验收；正式四头 PPO 已接入，recovery 暂缓。后续训练记录见 [P4](PGMT_P4_EXECUTION.zh-CN.md)。

## 计算口径

每个控制步使用冻结的实际承重状态 `s_f = 1[F_terrain,f,z > 10 N]` 与同一参考帧的左右脚标签。两脚的标签和地形力测量必须全部有效，且实际支撑模式与参考标签逐脚完全相等。多一只脚支撑或少一只脚支撑，都使整个环境的 T1 为 0。两脚均为摆动时也为 0。

持续匹配的支撑每步结算，不要求 touchdown 事件、历史年龄或最短支撑时长；重新落脚没有额外奖金。reset 后首个真实有效物理样本可以得分，reset 本身不计分。`touchdown` 事件定义仍保留，供调试使用。

实际 ankle-roll link 周围采样 **20×10 cm、3×3 点**，中心沿脚尖方向前移 **3.9 cm**。局部点为 `X={−0.061,0.039,0.139} m` 与 `Y={−0.05,0,0.05} m` 的笛卡尔积，随每只实际脚的 yaw 旋转，暂不使用 roll/pitch。通过 `PGMTTerrainImporter.query()` 获取当前位置的真实网格世界高度，保留负值；不从策略地图插值，不施加增强或归一化。

9 点须同时满足地图范围、射线命中/有限高度，以及当前 torso/yaw 的 **2×2 m 扫描几何覆盖**。这与随机 dropout 无关。blind 检查省略扫描器时仍保留相同几何范围；有扫描器时校验其配置，拒绝悄悄改变中心、方向、范围或 drift。

$$
\bar h_f=\frac{1}{9}\sum_{k=1}^{9}h_{f,k},\qquad
v_f=\frac{1}{9}\sum_{k=1}^{9}(h_{f,k}-\bar h_f)^2,
$$

$$
Q_f=\begin{cases}\exp(-v_f/0.03^2),&9\text{ 点全部有效},\\0,&\text{其他情况},\end{cases}
$$

令 `M=all(support_valid & reference_valid & (s == reference_contact))`，标签还必须为有限二值；`N_ref=sum(reference_valid & (reference_contact == 1))`：

$$
f_{\rm touchdown}=M\frac{\sum_f\mathbf1[c_f^{ref}=1]Q_f}{\max(1,N_{ref})},\qquad
r_{T1}=10\,\Delta t\,f_{\rm touchdown}.
$$

对参考支撑脚取平均，分母不随实际支撑脚数量或几何有效脚数量缩小。双支撑且一只脚的地图无效时最多得到一半；任一支撑状态不匹配则整体为 0。

数值核返回 `[0,1]` 的未加权率，RewardManager 仅乘权重 10 和 dt 各一次。当前 `dt=0.02 s`，单支撑、双支撑的**总上限均为 0.2/步**；双支撑两脚质量均为 1 时，各占 0.1。采用总体方差（分母 9），高度统一平移不改变结果。无效行先屏蔽再计算，不通过 `NaN×0` 避开无效数据。保留原始高度方差定义，因此坡度也影响质量。

## 生命周期和接口

- `PGMTState.rewards()` 在接触状态结算后，仅对整体匹配环境中的参考支撑脚批量查询，每脚 9 点。每个控制步使用当前真实脚位姿重新评价，不沿用旧落点质量；无合格支撑则不查询。
- `state.rewards()["terrain_touchdown_quality"]` 返回 `[N]` 的未加权、未积分率，原生适配器 `pgmt_reward(env, name="terrain_touchdown_quality")` 直接读取。
- `state.touchdown_state(for_reward=True)` 返回本控制步冻结的事件、9 点坐标/高度、地图/扫描/最终掩码、方差、单脚质量、总率及 motion/frame/episode 信息。新增 `eligible`、`pattern_valid`、`pattern_matched`、`reference_support_count`。`event` 仍表示真实 touchdown 上升沿，不再表示奖励门控。尚未结算当前步时拒绝返回旧数据。
- `state.touchdown_state()` 返回当前可读副本。局部 reset 清零指定行的事件/质量/率，令坐标/高度/方差为 NaN，其他环境保留；本步已结算的奖励快照不受 reset 影响。
- 不累计、不补发失配或无效期间的奖励；后续重新匹配/有效时只按当前步结算。重复读取不重新查询或更新事件。

形状：`query_xy_w=[N,2,9,2]`；`height_w/map_valid/scan_valid/point_valid=[N,2,9]`；`event/eligible/valid/variance/quality=[N,2]`；`rate/pattern_valid/pattern_matched/reference_support_count=[N]`。脚顺序左、右。未查询的 `valid=False` 不代表该处地形无效。旧 `evaluate_touchdown()` 函数保留为几何预览辅助函数，其原始两脚质量和不作为新 T1；运行期通过 `evaluate_support_quality()` 结算。

`pgmt_touchdown_quality.settlement` 随环境配置进入 checkpoint 的训练契约，防止把旧事件奖励下的优化器/价值目标当作相同任务静默续训。旧 policy 的推理加载不受影响；旧奖励 checkpoint 不能直接通过现有 strict resume，改变奖励后继续训练需要显式初始化/迁移。

代码：[数值核和配置](../pmt_tasks/pgmt_touchdown.py)、[运行期与缓存](../pmt_tasks/pgmt_runtime.py)、[独立奖励配置](../pmt_tasks/env_cfgs/pgmt/terrain_rewards.py)。

## 启用边界

Stage2 已默认计算并缓存 T1；**原 `pgmt_stage2` 基线的 RewardManager 保留 21 项/三组**，T1 不加入该基线的 PPO 总奖励。正式 `pgmt_stage2_terrain` 已在 P3.4 同步启用 terrain 组、四头与迁移，T1 归入 terrain，不归入 auxiliary 或 lower。

独立验收使用 `PGMTTouchdownRewardsCfg`：原 21 项加 T1，共 22 项、四组 `[upper,lower,terrain,auxiliary]`。真实 RewardManager 计算 T1 并验证分组总和，但不创建或训练四头策略，不改变指定 Stage1 checkpoint。

## 当前版本验收（2026-10-18）

- **259 passed**：在 Isaac 环境的 Python 3.12 / NumPy 2.3.1、CUDA 1 上完成 CPU/CUDA 回归，覆盖 T1、支撑/接触状态、参考标签、T2–T6、四头 PPO、checkpoint 及配置。日志：[unit_cpu_cuda.log](../logs/pgmt_t1_support_match_20261018/unit_cpu_cuda.log)。
- T1 覆盖双脚实际/参考支撑的 16 种组合、任一脚标签/力无效、非法标签、双摆动零奖励、单/双支撑同上限、无效地图不缩小分母、首次有效测量、稳定支撑、慢踱步不产生额外奖金、当前几何重查、终止快照和局部 reset。补充九点显示兼容测试后，最终 T1 CPU 集为 **53 passed / 2 CUDA skipped**，这两个 CUDA 案例已在上一项通过。日志：[t1_final_cpu.log](../logs/pgmt_t1_support_match_20261018/t1_final_cpu.log)。
- **Isaac passed**：CUDA 1、headless、5 环境、67 控制步，已自动退出。五类十级、两种脚 yaw 共 100 个几何窗口/900 点；动态支撑查询 3,780 点，包含 420 个得分脚步，其中 **412 个没有 touchdown 事件**，说明持续支撑正常结算；125 个模式失配的环境控制步均整体归零。
- 与原生 PhysX 射线独立比较，最大高度误差 **1.67e−6 m**，质量最大误差 **1.01e−5**。验证了 weight/dt 各一次、四组奖励和、同一步重复读取不重新查询、终止/局部 reset 保留旧快照、外部高台有全对象接触但无地形承重时 T1 为 0。
- 机器报告：[isaac_01/training_status.json](../logs/pgmt_t1_support_match_20261018/isaac_01/training_status.json)。没有启动 PPO、修改 checkpoint 或声称踱步已通过重新训练消除；8192 环境持续查询吞吐尚未单独测量。
- 环境说明：`pmt` 的 NumPy 1.x 无法读取旧 P2 的 NumPy 2 pickle，最初四头迁移回归因此报 `numpy._core` 缺失；改用兼容的 Isaac Python 环境后全部通过。该环境的无关 typeguard pytest 插件存在依赖冲突，验证时设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，未跳过上述测试或修改依赖。

当前 Isaac 验收命令：

```bash
bash scripts/validate_pgmt_contact_state.sh --touchdown --device cuda:1 \
  --log_dir "logs/pgmt_t1_support_match_$(date +%Y%m%d_%H%M%S)"
```

## 历史验证结果（事件版，2026-10-14）

- **93 passed**：CPU/CUDA 数值及接触状态/标签、原奖励、配置、课程、地图、参考回归。覆盖偏移/yaw、平地/负高度/斜坡/台阶、总体方差、无效点、扫描覆盖、异常位姿、事件稀疏查询、终止快照和不补发。
- **Isaac passed**：[机器报告](../logs/pgmt_p3_touchdown/isaac_02/training_status.json)。五类十级各一处、两种脚 yaw，共 **100 个几何窗口、900 点**；动态部分为 5 环境、67 个控制步、13 次真实落脚、117 点。两部分均用原生 PhysX raycast 独立核对，最大高度误差 **1.67e−6 m**，以 PhysX 高度和 NumPy 方差计算的质量最大误差 **1.01e−5**。几何探针不当作真实落脚事件。
- 真实 RewardManager 的 **weight/dt 只乘一次、四组和等于标量奖励、重复读取不重查、局部 reset、终止步快照、非地形高台不产生 T1** 均通过。对照中 640 个查询点高度为负，未裁成 0。
- 首次独立射线检查发现 headless 入口未开启 `enable_scene_query_support`，已修正验收入口后重跑通过；没有放宽高度误差容限。仿真自动退出。本次不包含 PPO 收敛或大规模吞吐验收。

```bash
bash scripts/validate_pgmt_contact_state.sh --touchdown \
  --log_dir logs/pgmt_p3_touchdown/$(date +%Y%m%d_%H%M%S)
```

`--touchdown` 只在此有限步入口启用 T1 RewardManager 项。入口为 headless，不启动 PPO；配置和报告写入指定目录。

## Isaac 九点交互检查

```bash
bash scripts/view_pgmt_contacts.sh \
  --log_dir logs/pgmt_p3_touchdown_viewer/$(date +%Y%m%d_%H%M%S) \
  --touchdown_points --pause_on_touchdown --speed 0.5 \
  --kit_args='--/exts/omni.syntheticdata/menubar/visible=false --/app/window/width=1600 --/app/window/height=1000'
```

加载 `model_15000.pt`，在 flat/L0 用真实物理回放步行动作；首次实际 touchdown 自动暂停，保留该步供检查。点击 `Pause / Run` 继续，`Step 20 ms` 单步，`Pause at next touchdown` 运行至下一次落脚；`Feet / overview` 切换脚部近景与全身视图。

- 每脚 9 个贴地球：绿色表示地图与扫描范围均有效；红色表示无效。黄色表示当前步实际触发 touchdown 的有效点。
- 为看清被脚模型遮挡的点，同 XY 位置另绘制一层高于查询高度 **25 cm** 的青色网格，以竖线连接贴地球。抬升仅影响显示；贴地球中心也只为防遮挡上移 12 mm，质量计算始终使用未偏移的真实查询高度。没有高度命中的点不伪造贴地球，只在诊断平面显示红色点。
- 面板按脚局部坐标显示 3×3 高度表：脚尖方向在上、+Y 在左，标注数值核的 1–9 点编号、有效点数、方差与 Q。另列每脚最近一次实际落脚的回合、帧号和质量。
- 连续显示包含摆动脚的几何预览；该额外查询只属于 viewer，不推进接触事件，不写回奖励缓存。合格支撑脚显示 `touchdown_state(for_reward=True)` 的实际结算几何；不合格脚保留明确标注的预览。面板显示整体匹配、有效性和本步 T1 rate。黄色仍表示 touchdown 事件，不代表额外奖励；reset 清理显示历史。

实现：[显示模块](../scripts/pgmt_touchdown_visualization.py)、[交互入口](../scripts/view_pgmt_contacts.py)。日志保存 `touchdown_points.json` 的逐点坐标/高度/掩码与最近落脚数据，自动暂停后保存 `touchdown_viewport.png`。

2026-10-14 GUI 实测：`logs/pgmt_p3_touchdown_viewer/live_01` 在控制步 209 / 参考帧 208 的右脚实际落脚处暂停；该脚 9/9 点有效、世界高度均为 0 m、总体方差 0、Q=1。已检查视口和完整窗口截图，确认脚底检测点、抬升网格、左右脚 3×3 数值及暂停/单步控件可见。随后按用户要求关闭平地窗口，切换到楼梯。

楼梯检查可在上述命令增加 `--terrain stairs --level 0 --patch_id 5 --spawn_xy 0.85 0`。选择 v6 的 L0、第 5 个 6×6 m 倒金字塔楼梯小块，初始位置相对小块中心沿 +X 偏移 0.85 m，置于台阶附近；`--yaw_deg` 控制初始朝向（默认 0）。高度仍由原生 reference placement 查询网格和全身碰撞几何后确定，机器人及参考共用变换，reset 重用所选位置。以上参数只作用于检查入口；默认仍为原来的 flat/L0，未修改训练出生策略。

楼梯 GUI 实测：`logs/pgmt_p3_touchdown_viewer/stairs_01` 在控制步 218 / 参考帧 217 的右脚 touchdown 暂停。该小块台阶高 0.056013 m、深 0.32 m；9 点全部有效，其中 8 点约 −0.224052 m，1 点约 −0.168039 m，跨越台阶边缘。实际奖励快照总体方差为 0.000309873 m²、Q=0.708714；已检查楼梯与九点视口截图，窗口保留供用户检查。
