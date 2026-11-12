# PGMT P2.1：PMT 感知语义对齐（version=3）

2026-10-12：按用户要求完成高度符号、无效高度设置、归一化位置和增强采样时机对齐。**147 项回归与 Isaac 实际验收通过**。旧 version=1/2 验收保留在 [历史记录](PGMT_P21_ACCEPTANCE.zh-CN.md)，不覆盖当前配置。

## 当前配置

| 项目 | 当前 PGMT |
| --- | --- |
| 环境 | `PGMTPerceptionEnvCfg`，v6 五类地形，21×21 扫描 |
| 扫描范围 | 2×2 m，0.1 m 间距，共 441 点；实际 torso 中心，yaw 对齐，row=y / col=x |
| 原生射线 | torso 上方 20 m 起射、向世界 −Z，最大距离仍为 60 m；起射偏移不参与高度零点 |
| 高度 | 与 PMT 相同：`sensor_world_z − ground_world_z − offset`，offset=0；地面低于 torso 为正 |
| 无效命中 | 地图外、未命中、非有限命中或随机丢点：**世界 Z 先填 30 m**，再参与高度变换、加噪及裁剪；mask=0，不再把最终值置 0 |
| sensor Z 异常 | 与 PMT 相同，NaN/±Inf 回退到 0.78 m |
| 随机丢点 | 每点独立概率 0.1 |
| 高度噪声 | 每点独立均匀分布 `U(−0.06,+0.06) m`，施加于相对高度，包括填充后的无效位置，顺序同 PMT |
| 观测裁剪 | `[-3,+3] m`；观测与 rollout 保存米制值，不除以 3 |
| 网络归一化 | `PGMTTerrainInput` 对输入再按 ±3 裁剪，然后除以 3，得到 `[-1,+1]`；mask 单独保持 0/1，无逐格运行均值归一化 |
| 增强采样时机 | **每次执行观测函数都重新采样**，与 PMT 一致；取消控制步/局部 reset 增强缓存 |
| 图与 mask 配对 | 一个 ObservationManager term 一次生成 `[N,882]`；wrapper 不采样，只拆成 `terrain_map[N,21,21]` 和 `terrain_valid[N,21,21]` |
| 有效区域 | 保留 PGMT 训练地图边界 mask；地图外命中按无效点走 PMT 填充流程 |
| 契约与续训 | version=3，完整配置写入课程/环境身份；version=1/2 或符号、单位、采样规则不匹配时拒绝续训 |

填充的是**世界 Z=30 m**，不是直接把相对高度设为 30、−3 或 0。例如 sensor Z=0.8 m 时，缺测得到 `0.8−30`，裁剪为 −3 m，网络归一化后为 −1；sensor Z=29.5 m 且关闭随机增强时，缺测得到 −0.5 m。两种情况均有测试覆盖。

`augmentation_enabled=False` 只关闭随机丢点和噪声，仍保持 PMT 的符号、填充和裁剪。实际几何核对继续读取 `terrain_scan()`：该历史干净接口为 ground−torso、无效 0，不加入增强，不作为 P2 策略图使用。

## 调用路径与范围

```text
原生 RayCaster hits + sensor pose + PGMT map bounds
  → PGMTState.terrain_observations()：每次调用重新采样
  → pgmt_terrain_observation()：一次打包高度和 mask
  → ObservationManager pgmt_terrain[N,882]
  → RslRlVecEnvWrapper：无随机拆分为 terrain_map / terrain_valid
  → rollout：保存本次米制图和 mask
  → 网络 PGMTTerrainInput：确定性裁剪 / 3，输出归一化图和 mask
  → 后续 P2.2 Terrain-Glimpse / P2.3 IFM（尚未接入）
```

每次新观测（包括同一步重复 `get_observations()`、局部 reset 后再次计算全量观测）会重新采样所有环境的噪声与丢点；局部 reset 不改变其他环境的真实几何。PPO 使用 storage 中保存的同一份输入，minibatch 只调用确定性网络预处理，不重新调用环境增强。

网络预处理层已实现并对照原 PMT 网络验证；**当前 Stage-1 actor 仍不读取地图**，未提前实现 selector、patch encoder 或地形 IFM，也没有执行 PPO 更新。本次没有更改原 PMT 代码或其扫描轴序。

| 代码 | 用途 |
| --- | --- |
| [pgmt_perception.py](../pmt_tasks/pgmt_perception.py) | version=3、世界高度填充、PMT 符号、每次调用重采样、原子打包/拆分 |
| [pgmt_runtime.py](../pmt_tasks/pgmt_runtime.py) | 从真实 sensor hits / pose 调用策略预处理，独立保留干净几何 |
| [stage2.py](../pmt_tasks/env_cfgs/pgmt/stage2.py) | 单一原子观测项，默认增强配置 |
| [vecenv_wrapper.py](../pmt_tasks/isaaclab_rl/rsl_rl/vecenv_wrapper.py) | reset / get_observations / step 三条入口共用无随机拆分；其他任务不受影响 |
| [pgmt_terrain_input.py](../motion_tracking_rl/networks/pgmt_terrain_input.py) | 网络侧裁剪与归一化，供 P2.2 接入 |
| [test_pgmt_perception.py](../tests/test_pgmt_perception.py) | 28 项专用验证，直接调用原 PMT 观测函数和网络预处理方法作对照 |

## 验收

2026-10-12 10:41:17–10:41:45，Isaac Sim 6 / CUDA:0 / headless，正常退出。

原始结果：[validation.json](../logs/pgmt_p21_acceptance/20261012_pmt_alignment_v3_r1/validation.json)；总记录：[acceptance_20261012_pmt_alignment_v3.json](../logs/pgmt_p21_acceptance/acceptance_20261012_pmt_alignment_v3.json)。

| 检查 | 结果 |
| --- | --- |
| PMT 全流程对照 | 真实仿真 hits/pose 和相同 CUDA RNG 下，PGMT wrapper 输出与原 `height_scan_for_vision()` 高度/掩膜逐值完全一致；不再做符号转换或无效清零来适配对照结果 |
| 地形与碰撞 | 800 小块、3,200 位姿、10,786 条独立 PhysX 射线；最大策略相对高度误差 5.007 μm；动态干净查询误差 4.768 μm |
| 丢点 | 52,920 个几何有效点中丢失 5,364 个，实测 **10.1361%** |
| 均匀噪声 | 47,556 个保留未饱和样本，范围 [−0.05999792,+0.05999941] m；均值 −0.00010477 m，标准差 0.03467480 m，符合理论值约 0.034641 m |
| 调用时机 | 同一步重复调用重新采样所有环境；局部 reset 后再次全量计算也重新采样，其他环境真实几何保持不变；原始 hits/干净扫描不受增强污染 |
| 无效点 | 地图边缘与随机丢点按 PMT 填充，当前场景中为 −3 m，网络侧为 −1；mask=0，全部有限 |
| 网络侧归一化 | 观测保留米制；独立网络输入层与原 PMT `_get_map_encoding()` 的高度/mask 通道逐值一致，且不消耗 RNG |
| Storage / minibatch | 24 真实控制步、120 份 wrapper 观测，2 轮 minibatch 共 240 次访问，所有观测逐值保留；网络重算不重采样 |
| 回归 | 全部 **147 项通过**，专用 **28 项**；Python 编译与 diff 检查通过 |

复现：

```bash
bash scripts/validate_pgmt_perception.sh --output logs/pgmt_p21_acceptance/new_v3_run
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt*.py tests/test_normalization_resume.py
```
