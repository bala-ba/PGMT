# PGMT P2.1 地形观测接入与验收

**当前状态（2026-10-12）：version=3 已完成 PMT 高度符号、无效填充、网络侧归一化及每次观测重采样的对齐；147 项回归和 Isaac 验收通过。当前配置与结果请看 [PMT 对齐记录](PGMT_P21_PMT_ALIGNMENT.zh-CN.md)。**

以下保留 2026-10-11 的 version=1/2 实现与历史验收。其 ground−torso 策略符号、无效清零、观测端归一化和按步缓存已被 version=3 替代，不能作为当前训练配置。首次 version=1 总报告：[acceptance_20261011.json](../logs/pgmt_p21_acceptance/acceptance_20261011.json)；历史原始结果：[validation.json](../logs/pgmt_p21_acceptance/20261011_r2/validation.json)。

## version=2 实现（历史）

| 内容 | 代码 |
| --- | --- |
| 独立 P2 感知环境 | [stage2.py](../pmt_tasks/env_cfgs/pgmt/stage2.py)：`PGMTPerceptionEnvCfg` 继承已验收扫描配置，默认 v6 / clipped 动作库 / 随机 XY/yaw / 固定第 0 帧，保留原三组奖励，碰撞栈 256 MiB |
| 感知输入契约 | [pgmt_perception.py](../pmt_tasks/pgmt_perception.py)：version=2；独立 `terrain_map` / `terrain_valid`，float32 `[N,21,21]`；2×2 m、0.1 m；实际 torso yaw 坐标、row=y/col=x。原始高度仍为地面减 torso；默认策略值为 `clip(h+U(-0.06,0.06),-3,3)/3`，无量纲，无效处最终置 0 |
| 增强与缓存 | 同文件 `TerrainObservationCache`：默认 10% 独立丢点、±0.06 m 均匀噪声，裁剪/固定尺度归一化；地图和掩膜同一步重复读取一致，局部 reset 只刷新所选环境；旧快照及干净数据不修改 |
| 运行期入口 | [pgmt_runtime.py](../pmt_tasks/pgmt_runtime.py)：`terrain_scan()` 保留干净几何；新增 `terrain_observations()` 返回策略副本；reset 传入 env_ids 使噪声局部失效 |
| ObservationManager | [pgmt_observations.py](../pmt_tasks/mdp/pgmt_observations.py)：地图/掩码读取同一策略缓存；`terrain_valid=geometry_valid & ~dropout`，干净 `terrain_scan()["valid"]` 不含丢点 |
| 保存身份 | [pgmt_curriculum.py](../pmt_tasks/pgmt_curriculum.py)：P2 环境身份包含完整感知契约，沿用已有课程 checkpoint 的严格身份比较。旧 P1 未配置感知契约，身份格式保持原样 |

复用原生 RayCaster 与完整 PGMT 网格。独立环境配置已实现；Gym/task YAML、感知网络注册与完整训练入口仍按计划在 P2.4 接入。

使用示例（Isaac App 启动后）：

```python
from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
from pmt_tasks.pgmt_perception import terrain_observation_contract
from pmt_tasks.pgmt_runtime import get_pgmt_state

cfg = PGMTPerceptionEnvCfg()  # 默认 PMT 增强：10% 丢点，±6 cm 均匀噪声，裁剪并除以 3
# 仅几何核对 / 扫描 GUI 使用以下覆盖，默认训练不要关闭：
# cfg.pgmt_terrain_observation = terrain_observation_contract(augmentation_enabled=False)

# env 构建完成后：
scan = get_pgmt_state(env).terrain_scan()           # 干净米制图和几何 mask
policy_map = get_pgmt_state(env).terrain_observations()  # 增强且归一化的图、含丢点的 mask
# wrapper.get_observations()["terrain_map" / "terrain_valid"] 供训练 storage 保存
```

无效点始终为有限的 0 占位并伴随 `valid=False`，不是零高度地面。干净扫描完整保留负高度和凹地；策略图按上述公式转换且不截负值为 0。原 PMT 的世界高度 `30 m` 无效填充值不复制到 PGMT，保留已有 0+mask 约定；高度符号仍为 ground−torso。增强分布、概率、裁剪和尺度与原 PMT 视觉配置一致，坐标和无效编码差异明确保留。

## PMT 增强追加验收（version=2，历史）

2026-10-11 17:42:30–17:42:55，Isaac Sim 6 / CUDA:0 / headless 验收通过，进程正常退出。总记录：[acceptance_20261011_pmt_augmentation.json](../logs/pgmt_p21_acceptance/acceptance_20261011_pmt_augmentation.json)；原始结果：[validation.json](../logs/pgmt_p21_acceptance/20261011_pmt_augmentation_r1/validation.json)。未执行 PPO 更新。

对齐来源为 [原 PMT VisionCfg](../pmt_tasks/env_cfgs/pmt/distill_stepping_stone.py)、[height_scan_for_vision](../pmt_tasks/mdp/observations.py) 和 [VisionTransformerActorCritic](../motion_tracking_rl/networks/vision_transformer_actor_critic.py) 的 ±3 m 裁剪与除以 3。`enable_corruption=False` 不关闭函数内部增强；PGMT 同样由观测契约显式控制。

| 检查 | 结果 |
| --- | --- |
| 原 PMT 数值对照 | 测试直接加载原 `height_scan_for_vision` 及其三个纯函数，给定相同相对高度和随机种子；在归一化及 PGMT 无效清零后，地图和丢点 mask **逐值完全一致**，覆盖正负高度、饱和、真实零值和原有缺测 |
| 几何复验 | 800 小块、3,200 位姿、1,433,250 扫描单元；10,786 条独立 PhysX 射线；最大碰撞高度误差 **5.007 μm**，GPU 网格查询最大差 **6.676 μm** |
| 丢点统计 | 24 控制步、52,920 个几何有效点，丢失 **5,215** 个，实测 **9.8545%**；地图外点始终无效，无效值全部为 0 |
| 均匀噪声统计 | 47,705 个保留且未饱和点，转换回米后噪声范围 **[−0.05999792, +0.05999929] m**；均值 **−0.00014413 m**，标准差 **0.03454631 m**，符合 `U(−0.06,0.06)` 理论标准差约 0.034641 m |
| 裁剪与单位 | 默认策略值在 `[-1,1]`；±3 m 饱和、负值和零值有独立边界测试；原始几何查询仍为 m。几何模式关闭整条增强/预处理后，图和 mask 与干净扫描逐值相同 |
| 原始数据与缓存 | 原始 hits / 干净高度 / 几何 mask 不变；重复读取不重采样；局部 reset 同时只刷新所选环境的图和 mask，其他环境及旧快照不变 |
| Rollout / minibatch | 120 份真实 wrapper 观测、两轮打乱共 240 次访问，所有观测组逐值一致；更新阶段未重采样增强 |
| 配置与续训身份 | 默认增强开启；配置实例独立；契约进入环境身份，配置变化使身份变化；version=1 和错误单位/裁剪规则明确拒绝 |
| 回归 | 全部 **143 项通过**，其中增强专用 **24 项**；Python 编译及 diff 检查通过 |

[pmt_augmented_maps_L9.npz](../logs/pgmt_p21_acceptance/20261011_pmt_augmentation_r1/pmt_augmented_maps_L9.npz) 保存五类 L9 的实际策略图、策略 mask、干净米制图和几何 mask；同目录保存干净 L0/L5/L9 碰撞对照图，以及默认增强/干净模式的两份契约。

GUI 扫描检查器显式使用 `augmentation_enabled=False`，界面标注 `clean metric scan (augmentation OFF)`；其用途是核对真实表面，不能据此判断默认训练关闭了增强。后续网络直接接收已归一化的策略图，不再重复归一化。

P2.2 计划已同步修正：10% 独立丢点会让“25 点全有效”的保留概率降至约 7.18%，因此改用显式 mask 输入、掩膜加权插值和有效覆盖率判据；该网络尚未实施，详细维度及待验收项见 [P2 计划](PGMT_P2_PLAN.zh-CN.md)。

## 首次验收结果（version=1，历史）

运行环境：Isaac Sim 6 + PMT Isaac Lab，RTX 5090 / CUDA:0，headless，5 个环境，每类一个。v6 地形 SHA-256：`5fc8e5e577feff19adfdc29bfeffe6bc2dc6736dbb4a54a8134d562c73bb665d`。

| 检查 | 覆盖与结果 |
| --- | --- |
| 完整网格与坐标 | 五类×十级×16 小块×四个 yaw，共 **3,200 个传感器位姿**；含边缘/刷新检查，共 **1,433,250 个扫描单元**。地图轴序、旋转、负高度均通过 |
| GPU 网格查询 | 扫描与按实际世界 XY 查询的高度最大差 **6.676 μm** |
| 独立 PhysX 碰撞 | 全地图抽查 **4,171 条**独立射线；另在 L0/L5/L9 五类上逐格重建完整 21×21 对照图，共 **6,615 条**；合计 **10,786 条**，最大相对高度误差 **4.768 μm**，小于预设 `1e-4 m` 容差 |
| 跨区域与地图边缘 | 跨小块/等级/样式查询通过；出生分配不变；四侧边缘各有 42 个正确无效点；仅扫描越界不终止，机器人越界正确 truncation 并刷新地图 |
| 噪声隔离 | 设 `std=0.01 m`，原始 sensor hits 和干净扫描逐值不变；无效点保持 0。52,920 个有效样本的实测均值 **−0.00001952 m**、标准差 **0.0099668 m** |
| 同步与局部 reset | 同步重复读取地图/掩码完全一致；reset 环境 0 刷新其输入，其余环境地图逐值不变；切回无噪声后与干净地图完全一致 |
| 动态扫描 | 追加 **24 个真实控制步**，每步核对当前 torso 位姿与网格查询，最大干净高度误差 **4.768 μm**；加之前边界检查共 26 个控制步 |
| 原生 storage / minibatch | 实际 wrapper 观测形成 **120 个样本**，经过 `RolloutStorage.add_transitions()` 与两轮打乱 minibatch，共 **240 次样本访问**，所有观测组逐值一致，未重新采样噪声 |
| 配置与元数据 | 感知配置实例互不共享可变字典；环境身份含观测契约，噪声配置变化对应不同身份；默认 v6 与扫描器启用正确 |
| 自动化回归 | `tests/test_pgmt*.py` 与 normalizer 恢复测试共 **131 项通过**；Python/shell 语法与 diff 检查通过 |

独立 PhysX 完整对照图：每行一个地形，左列为 `terrain_map`，中列为真实碰撞高度减 torso 高度，右列为绝对误差。

- [L0 对照图](../logs/pgmt_p21_acceptance/20261011_r2/terrain_map_L0.png)
- [L5 对照图](../logs/pgmt_p21_acceptance/20261011_r2/terrain_map_L5.png)
- [L9 对照图](../logs/pgmt_p21_acceptance/20261011_r2/terrain_map_L9.png)

同目录 `.npz` 保存观测地图、PhysX 高度、有效掩码、误差、实际世界采样 XY、地面世界 Z、torso 位姿和地形名称，可独立复查。

## 问题与范围

首轮 `20261011_r1` 已通过全地图与完整 PhysX 对照，后续脚本在 inference-mode 局部 reset 之后切回普通上下文执行 `env.reset()`，触发原生 command metrics 的 inference tensor 原地写限制。已让追加验收遵循 runner 的统一 `torch.inference_mode()` 调用上下文，`20261011_r2` 完整重跑通过并正常关闭；首轮失败结果保留，不计入通过结果。成功日志无 Traceback、PhysX 碰撞栈溢出或丢弃接触记录。

这是感知输入验收：静态部分使用受控传感器位姿，动态部分使用单个 walk 片段并关闭跟踪终止及课程更新，只保留地图越界。原生 storage 的动作/价值字段为验收占位，不执行 PPO 更新。全库初始化/课程/盲走 PPO 的证据仍属于 P1；P2.2 的 glimpse 和 P2.3 的 IFM 尚未接入。

## 复现

```bash
bash scripts/validate_pgmt_perception.sh \
  --output logs/pgmt_p21_acceptance/new_run

/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt*.py tests/test_normalization_resume.py
```

验收入口复用 `validate_pgmt_v6.sh` 的 Sim 6 运行环境；输出目录需要使用新的路径，以保留历史结果。
