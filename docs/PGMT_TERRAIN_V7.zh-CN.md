# Stage2 v7：PBFM 七类地形与训练比例

本页记录历史 v7 近似实现；当前默认已切换为直接转换 P-BFM 原生几何的 [v8](PGMT_TERRAIN_V8.zh-CN.md)。
v7 使用
[`configs/terrain/pgmt_stage2_v7.yaml`](../configs/terrain/pgmt_stage2_v7.yaml) 和
`assets/terrain/pgmt_stage2_v7/`，依据为 [`pbfm_terrain.md`](pbfm_terrain.md)。
v5/v6 的配置和已生成资产保留，v7 是单独生成的版本。

## 地形及比例

| 文档地形 | v7 family | 训练比例 | 随难度变化的参数 |
|---|---|---:|---|
| Flat | `flat` | 20% | 平坦地面 |
| Perlin Rough | `rough` | 10% | 高度幅度 A：0.02–0.06 m |
| Low Stairs Up | `stairs_up` | 20% | 金字塔台阶单级高度：0.10–0.20 m |
| Low Stairs Down | `stairs_down` | 20% | 倒金字塔台阶单级高度：0.10–0.20 m |
| Low Platforms | `platforms` | 10% | 随机网格高度幅度 H：0.03–0.10 m |
| HF Pyramid Slope Inv | `slope` | 10% | 坡度 rise/run：0.0–0.2，最大坡角约 11.31° |
| Boxes | `boxes` | 10% | 离散障碍物高度：0.05–0.15 m |

`stairs_up` 中心高、向四周降低；`stairs_down` 中心低、向四周升高。
v6 的 `stairs` 对应 v7 的 `stairs_down`；v6 的 `boxes` 是随机网格，对应 v7 的 `platforms`。
v7 的 `boxes` 现在专指离散障碍物，不能直接沿用 v6 同名列解释结果。

比例用于**并行训练环境分配**，不通过地图面积近似。每个地形占一个等宽地图列，
Importer 按 `family_proportions` 的最大余数法分配环境，保证总数准确，
每类数量与 `比例 × 环境数` 的差小于 1。160 环境分配为
`[32, 16, 32, 32, 16, 16, 16]`。每个训练进程独立分配。
课程升降级只改变 level，不改变 family，因此每步采样比例保持稳定；
不同地形 episode 长度可能不同，完成 episode 的数量比例不必相同。
TensorBoard 的 `TerrainMix/<family>/target_fraction` 和 `actual_fraction` 可核对配置与实际值。

## L0–L9 参数

保留原先的固定种子 `1731` 和分段采样：每级 `d=(L+u)/10`，
`u` 来自按级固定的随机序列；同一级所有地形共享 d。
各参数在上述区间内按 d 线性插值，**坡面线性插值的是斜率，不是角度**。
因此实际 L0/L9 不恰好取区间端点，生成时另外检查 d=0、1。

| 等级 | d | Perlin 幅度 A（cm） | 上/下台阶单级高度（cm） | 网格幅度 H（cm） | 斜率 / 坡角 | 障碍物高度（cm） |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0.0241 | 2.10 | 10.24 | 3.17 | 0.0048 / 0.28° | 5.24 |
| L1 | 0.1957 | 2.78 | 11.96 | 4.37 | 0.0391 / 2.24° | 6.96 |
| L2 | 0.2898 | 3.16 | 12.90 | 5.03 | 0.0580 / 3.32° | 7.90 |
| L3 | 0.3524 | 3.41 | 13.52 | 5.47 | 0.0705 / 4.03° | 8.52 |
| L4 | 0.4307 | 3.72 | 14.31 | 6.01 | 0.0861 / 4.92° | 9.31 |
| L5 | 0.5639 | 4.26 | 15.64 | 6.95 | 0.1128 / 6.44° | 10.64 |
| L6 | 0.6827 | 4.73 | 16.83 | 7.78 | 0.1365 / 7.77° | 11.83 |
| L7 | 0.7075 | 4.83 | 17.08 | 7.95 | 0.1415 / 8.05° | 12.08 |
| L8 | 0.8656 | 5.46 | 18.66 | 9.06 | 0.1731 / 9.82° | 13.66 |
| L9 | 0.9530 | 5.81 | 19.53 | 9.67 | 0.1906 / 10.79° | 14.53 |

表中为生成器输入值。高度场按 0.005 m 量化；离散障碍物沿用 Isaac Lab 向下取整，
例如 L0 实际顶高 0.05 m、L9 为 0.145 m。网格顶面在 `[-H,+H]` 内采样。
Perlin 为有正负变化的梯度噪声，每块归一化后乘 A，再做高度量化。

## 文档未指定的参数

- 保留 6×6 m 地块、4×4 地块组成一个 24×24 m 区域；现在为 70 个区域、1,120 个地块，
  有效地图 240×168 m，外侧边带 20 m。
- 上下楼梯踏深均为 0.32 m，中央平台宽 2 m，边带 1 m。
- 倒金字塔坡面中央平台参数 0.5 m，边带 0.25 m，水平采样间距 0.10 m。
- 网格宽 0.70 m，8×8 网格，无中央平台；地块间连接带总宽 0.40 m。
- Perlin 使用本地实现的梯度噪声，经 Isaac Lab 原生高度场转换器生成 mesh：
  基础波长 1.20 m、3 个 octave、persistence=0.5、lacunarity=2，水平采样间距 0.15 m。
  平地使用本地 Isaac Lab 的 `MeshPlaneTerrainCfg` 提供等价平坦支撑面。
- 离散障碍物使用原生 `HfDiscreteObstaclesTerrainCfg`，每块尝试放置 20 个正高度障碍物，
  `obstacle_height_mode=fixed`；宽度配置 `[0.40,0.80)` m，原生离散步长为 4 个水平采样点，
  当前实际宽度为 0.40 m；允许障碍物重叠。无中央平台，边带 0.25 m。
  关闭陡面顶点移动（`slope_threshold=null`），保持单值高度场；边缘通过 0.10 m 网格过渡，
  避免相邻障碍物拐角的折叠三角形导致 PhysX 与扫描不一致。
- 初始课程等级 L0–L1、中央 2×2 地块出生、随机 XY/yaw、参考起始帧 0 保持现有设置。

上下楼梯继续映射到动作兼容表中的 stairs 语义组；网格平台和离散障碍物都使用 boxes 组。
原有 `up_stairs` 排除标签仍只排除倒金字塔变体。下肢 tracking relaxation 继续按语义组工作。
这些映射没有更改手工动作兼容 CSV。

## 重建与检查

```bash
# 在已配置 Isaac Lab 的 pmt 环境中；已有 manifest 的目录禁止直接覆盖。
python scripts/build_pgmt_patch_terrain.py --headless --device cpu

# Isaac Sim 6 环境，验证所有采样点的实际 PhysX 碰撞高度。
python scripts/view_pgmt_patch_terrain.py --headless --validate-only --gpu 0

# 交互查看新地形。
python scripts/view_pgmt_patch_terrain.py --family stairs_up --level 9 --view patch
```

构建器默认读取 v7；指定 `--config configs/terrain/pgmt_stage2_v6.yaml --output <新目录>`
仍可重建原 v6。生成的 USD/PLY/manifest 延续现有仓库规则，不纳入 Git。
v7 配置、生成代码及本说明纳入版本管理。

新训练默认使用 v7；旧 checkpoint 的地形、课程状态和新版本不一致，不能作为 v7 的严格续训。
旧策略回放默认使用 checkpoint 中记录的资产路径；交互回放可用 `--terrain_asset` 显式覆盖。
Stage1→Stage2 的现有初始化流程继续可用；本次不启动正式 PPO 训练。

## 本次验证（2026-11-11）

- 70 个区域、1,120 个地块及全部地形 d=0/1 端点的几何检查通过。
- Isaac Sim 6 / PhysX 共 390,996 次射线检查通过，最大高度误差 `6.997e-6 m`，
  地块接缝误差为 0。报告：`logs/pgmt_terrain_preview_v7/collision_validation.json`。
- Isaac Lab / Sim 5.1 的 CPU 训练环境中，10 个机器人实际分配为 `[2,1,2,2,1,1,1]`；
  L0、L5、L9 全部完成重置及短步仿真，检查出生间隙、动作兼容、中央出生地块、有限奖励和高度扫描。
  每级 4,410 个扫描格均有效。报告：`logs/pgmt_v7_runtime_smoke.json`。
- 配置、比例、Perlin、语义映射、课程、旧 v6 布局、跟踪放宽及回放相关测试：
  114 通过、1 项因 CUDA 不可用跳过。本次验证未运行 GPU PPO 训练或收敛评测。
