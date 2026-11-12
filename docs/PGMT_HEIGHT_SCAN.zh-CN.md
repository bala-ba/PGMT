# PGMT 地形扫描接口

`PGMTPlacementEnvCfg` 已接入 PMT 原生 RayCaster 的网格、yaw 对齐、更新与 reset 机制。Stage 1 配置保持无扫描器。P2.1 已新增独立 `PGMTPerceptionEnvCfg`、感知契约及观测副本噪声，见 [接入与验收](PGMT_P21_ACCEPTANCE.zh-CN.md)；Terrain-Glimpse、IFM 仍待后续实现。

## 扫描约定

| 项目 | 当前实现 |
| --- | --- |
| 中心和朝向 | 实际机器人 `torso_link` 中心，只跟随其 yaw，不跟随 roll/pitch；不使用参考或出生 tile 的坐标系 |
| 网格 | 2 m × 2 m，0.1 m 间距，含端点的 21 × 21 点，共 441 条射线 |
| 轴序 | `map[row=y, col=x]`；行、列坐标分别从 −1 m 到 +1 m；中心为 `[10,10]`；正 x 为 torso 前方、正 y 为左侧 |
| 高程 | `height = ground_world_z - torso_world_z`，单位 m；地面低于 torso 为负，高于 torso 为正 |
| 射线 | torso 上方 20 m 起射，沿世界 −Z 向下，最大距离 60 m；起射偏移不参与观测零点 |
| 更新 | 每控制步更新，`update_period = decimation × sim.dt`，当前 0.02 s；reset 后重新查询 |
| 有效性 | 射线命中坐标与 torso 高度有限，且命中 XY 在有效训练地图内；不包含外围 border |
| 跨 tile | 每点按当前世界位置判断有效性，能覆盖多个 tile；不修改出生课程等级、地形类型和 origin |
| 越界 | 仅扫描点越界不触发 done；当前机器人或参考越界仍走已有 `map_out_of_bounds` truncation |

PGMT USD 包含多块独立网格（v5 为 70 块，v6 为 800 个小块），原生 RayCaster 的默认发现逻辑只取路径下第一个 Mesh。因此 `PGMTHeightScanner` 仅替换网格发现步骤，复用 `PGMTTerrainImporter.query_mesh` 的完整世界坐标网格，其余仍采用原生 RayCaster。扫描与 reset/奖励干净查询共享同一组已校验三角面，不重复创建 GPU 网格。注册使用弱引用，实际网格由 importer 持有，扫描器初始化后按原生缓存生命周期管理。

2026-10-11 已补齐 v6 验收：800 小块、四个 yaw、3,200 传感器位姿，以及跨小块/等级/样式、四侧边界和 reset 刷新均通过。完整结果见 [v6 P1 补充验收](PGMT_V6_P1_ACCEPTANCE.zh-CN.md)。盲走训练继续关闭扫描器。

## 训练接口

P2 ObservationManager 单次计算原子观测 `pgmt_terrain[N,882]`（441 个高度 + 441 个有效性值），保证一次丢点采样对应一份高度和 mask。RSL-RL wrapper 只拆分、不重新采样，训练接口仍返回两个张量：

```python
obs["terrain_map"]    # float32 [N,21,21]，sensor_z-ground_z，加噪并裁剪 [-3,3]，单位 m
obs["terrain_valid"]  # float32 [N,21,21]，1=几何有效且未被随机丢弃
```

训练期干净接口：

```python
from pmt_tasks.pgmt_runtime import get_pgmt_state

scan = get_pgmt_state(env).terrain_scan()
scan["height"]    # float32 [N,21,21]，干净地面减 torso 高度，单位 m
scan["valid"]     # bool [N,21,21]，真实几何有效性，不含随机丢点
scan["height_w"]  # float32 [N,21,21]，带符号世界高度；无效为 NaN，仅用于诊断
```

**2026-10-12：`PGMTPerceptionEnvCfg` 使用 version=3，按用户要求对齐 PMT 的符号、无效填充、归一化位置和采样时机。** 先将地图外/未命中/随机丢弃点的世界 Z 填为 30 m，sensor Z 非有限时回退到 0.78 m；计算 `sensor_z - ground_z - offset`（offset=0），加均匀噪声 ±0.06 m 并裁剪到 ±3 m。随机丢点概率为 10%。无效点不再清零，在当前场景通常为 −3 m，同时 `terrain_valid=0`；这不是实际地面高度。

**观测和 rollout 都保留米制值。** 网络输入层 `motion_tracking_rl/networks/pgmt_terrain_input.py::PGMTTerrainInput` 按 PMT 逻辑裁剪后除以 3，保持 mask 为 0/1；无效网络高度通常为 −1。P2.2 负责在 selector/patch 前调用该层，当前 Stage-1 actor 尚未接入感知。

增强不再按物理控制步或局部 reset 缓存：**每次调用观测函数都为全部环境重新采样**。同一步多次 `get_observations()` 也会变化；局部 reset 不改变其他环境的几何，但下一次全量观测会重新生成所有环境的噪声/丢点。一次返回中的图和 mask 始终配对，rollout/minibatch 保存这一次采样；网络更新不重新请求环境观测，也不重新增强。

sensor hits 和 `terrain_scan()` 的干净米制几何仍不修改。上表“扫描约定”及 `terrain_scan()["height"]` 保留已有 ground−torso 语义和无效 0 占位，供几何/奖励核对；**P2 策略观测采用上述 PMT 语义**。原 `PGMTPlacementEnvCfg` 保持旧几何观测接口。`augmentation_enabled=False` 现在只关闭随机噪声与丢点，不关闭 PMT 符号、填充或裁剪；GUI 显示实际干净表面，并额外验证无噪声 PMT 策略图。version=1/2 契约不允许作为 version=3 静默续训。完整配置与验收见 [PMT 对齐记录](PGMT_P21_PMT_ALIGNMENT.zh-CN.md)。

新增观测已经通过 RSL-RL wrapper 传递，但现有 Stage 1 actor 不读取它们；需要完成后续 Terrain-Glimpse/IFM 后，策略才会利用地形输入。原有 `configs/sensor/height_scan.yaml` 是其他任务的模板，此接口的有效扫描配置位于 `pmt_tasks/env_cfgs/pgmt/placement.py`。

## 验证入口

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_height_scan.py tests/test_pgmt_terrain_query.py \
  tests/test_pgmt_reference.py tests/test_pgmt_mdp.py tests/test_pgmt_config.py

OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/validate_pgmt_height_scan.py --headless --device cuda:0
```

集成验证覆盖七种地形、十级难度、四个 yaw 下的 280 次出生放置，检查所有扫描点的轴序、朝向和高度；使用独立 PhysX 地形碰撞查询交叉核对。另检查负高度凹地、跨等级/样式边界、扫描越界掩码、机器人越界截断、reset 刷新和 wrapper 输出。验证中的人为位移关闭了其他跟踪终止项及课程更新，不代表策略跟踪性能。结果写入 `logs/pgmt_height_scan/validation.json`，以其中 `passed` 为准。

2026-10-10 验证通过：30 项单元/回归测试；Isaac 280 次出生放置、350 次独立 PhysX 查询和 2 个真实控制步。扫描与干净网格查询最大高度误差约 `5.72e-6 m`，与 PhysX 最大误差约 `4.29e-6 m`；地图边缘案例正确屏蔽 94 个点，扫描越界未触发截断，机器人越界触发截断并返回新出生位置的扫描。
