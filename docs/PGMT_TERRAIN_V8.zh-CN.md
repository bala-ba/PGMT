# Stage2 v8：直接转换 P-BFM 原生地形

`pgmt_stage2` 和 `pgmt_stage2_terrain` 当前默认使用
[`pgmt_stage2_v8.yaml`](../configs/terrain/pgmt_stage2_v8.yaml)，资产位于
`assets/terrain/pgmt_stage2_v8/`。v5/v6/v7 的已有资产保留。

本次直接调用兄弟仓库 `../P-BFM/pbfm/terrains/presets.py` 的
`make_training_mix_generator_cfg()`、`add_training_nonflat_guard_ring()` 和
`add_training_outer_walls()`，使用 MJLab 1.4.0 / MuJoCo 3.8.0 编译，
再把原始碰撞盒体和高度场转换为 Isaac 的三角网格。高度场保留原网格点、
三角形对角线、底面和侧壁；没有重新采样地形高度。

来源仓库 HEAD 为 `ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7`。
资产 `manifest.json` 还记录实际读取的 P-BFM/MJLab 源文件 SHA256、完整原始配置、
每个地块的实际 difficulty、PLY/USD 校验值和原生 MuJoCo 高度探针；
`source.mjb` 保留本次编译的原始碰撞模型。

## 地图与训练比例

- 10 行难度 × 7 列地形，每格 5×5 m，共 70 个训练地块；每个 region 只有 `P00`。
- 核心地图为 50×35 m，XY 范围 `[-25,25] × [-17.5,17.5]`。
- 地块边缘各保留 0.5 m 平地，相邻非平坦区域之间形成 1 m 平坦通道。
- 核心外有两圈、共 84 个 5×5 m 保护地块，Perlin 与离散障碍物交替；
  内部抬高 0.05 m、边缘仍为零，原版外围平面底托也完整保留。
- 保护区再向外是 2 m 平坦安全带，外墙高 5 m、厚 0.05 m。
  墙内可行走范围为 `[-37,37] × [-29.5,29.5]`。
- 训练出生在核心地块；保护区和安全带参与碰撞、扫描、接触力和有效地图判断，
  不分配课程 family/level，不作为随机出生地块。

| 地形 | PMT family | 环境比例 | 原版参数 |
|---|---|---:|---|
| 平地 | `flat` | 20% | z=0 |
| Perlin 粗糙地面 | `rough` | 10% | 高度参数 0.02–0.06 m；2 层、persistence=0.25、lacunarity=2、scale=20；分辨率/水平步长 0.2 m |
| 金字塔楼梯 | `stairs_up` | 20% | 5 级；单级高度 0.10–0.20 m；踏面宽 0.30 m；中心平台 1.2 m |
| 倒金字塔楼梯 | `stairs_down` | 20% | 同上，中心低、四周高 |
| 随机网格平台 | `platforms` | 10% | 网格宽 0.8 m；高度幅度 0.03–0.10 m；中心平台 1.2 m；相近高度合并阈值 0.025 m，最大合并距离 3 |
| 倒金字塔坡面 | `slope` | 10% | rise/run 为 0–0.2；中心平台 1.2 m；水平步长 0.2 m |
| 离散障碍物 | `boxes` | 10% | 50 次矩形区域写入；H=0.05–0.15 m；高度从 `-H,-H/2,H/2,H` 选择；宽、长独立从 0.4/0.6/0.8/1.0 m 选择；中心平台 1.2 m |

离散障碍物是带有凸起和凹坑的高度场。50 次写入允许相互覆盖，
因此最终可见障碍物数不一定等于 50。其声明宽度范围为 0.3–1.0 m，
经过原版 0.2 m 网格取整后的采样集合为上述四个值；高度量化步长为 0.005 m。

环境比例通过 `family_proportions` 的最大余数法分配，课程只升降 level、保持 family。
例如 160 个并行环境分配为 `[32,16,32,32,16,16,16]`；地图七列仍然等宽。
TensorBoard 的 `TerrainMix/<family>/target_fraction` 和 `actual_fraction` 可核对实际分配。

## L0–L9 的实际参数

沿用原版 seed=0 和共享随机数生成顺序。每格的
`difficulty=(level+随机数)/10`，不同 family 在同一级的 difficulty 也不同。
下表为生成配置插值后的参数；单位均为 m，坡度列为无量纲 rise/run。
高度场量化、归一化、平台和边界处理后的实际表面以资产网格及原生射线结果为准。

| Level | Perlin 高度参数 | 上楼梯单级高 | 下楼梯单级高 | 网格幅度 | 坡度 | 障碍物 H |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0.02326 | 0.10004 | 0.10080 | 0.03599 | 0.01780 | 0.05781 |
| L1 | 0.02467 | 0.11732 | 0.11783 | 0.04125 | 0.02686 | 0.06607 |
| L2 | 0.03090 | 0.12369 | 0.12572 | 0.04469 | 0.05130 | 0.07185 |
| L3 | 0.03306 | 0.13263 | 0.13077 | 0.05680 | 0.06261 | 0.08665 |
| L4 | 0.03826 | 0.14951 | 0.14977 | 0.06304 | 0.08105 | 0.09008 |
| L5 | 0.04070 | 0.15022 | 0.15111 | 0.06627 | 0.11680 | 0.10079 |
| L6 | 0.04587 | 0.16627 | 0.16492 | 0.07596 | 0.12258 | 0.11855 |
| L7 | 0.05018 | 0.17018 | 0.17031 | 0.08118 | 0.14271 | 0.12159 |
| L8 | 0.05590 | 0.18382 | 0.18406 | 0.08792 | 0.17314 | 0.13992 |
| L9 | 0.05613 | 0.19313 | 0.19492 | 0.09414 | 0.18430 | 0.14868 |

PMT Stage2 采用 **地块中心 XY + 随机 yaw** 初始化，对齐 P-BFM 的位置和朝向采样方式。
`motion.placement_xy_mode: center` 将根部 XY 固定在所选地块的几何中心，
yaw 在 `[-π, π]` 内均匀提议采样，机器人与参考轨迹一起旋转。
碰撞放置检查只调整高度；重试时 XY 仍为中心，并重新采样 yaw。
PMT 保留原有动作兼容规则和全身碰撞放置检查。2026-11-11 起，Stage2 已进一步采用
P-BFM 的动作级 EMD 权重与片段内均匀随机起点；地形课程按剩余参考完成情况结算。
配置与评估流程见 [EMD 采样说明](PGMT_EMD_SAMPLING.zh-CN.md)。仍不引入躺倒采样。
出生 patch 集合随资产读取：v8 为 `[0]`，旧 v6/v7 仍从中心四块选择一个，再使用该小块中心。
如需显式恢复地块内随机 XY，可覆盖 `motion.placement_xy_mode=random`；
Stage1 和旧 P1 的默认初始化保持原逻辑。中心出生模式记录在训练/课程契约中，
不会将采用随机 XY 的旧 checkpoint 当成相同配置严格续训。
静态地形预览里的黄色点仍是资产生成时的表面候选探针，并非当前机器人的出生位置。
地形几何来自 P-BFM；机器人、奖励、PhysX 接触材料和训练算法仍使用 PMT。
本次没有启动 PPO 训练。旧 checkpoint 回放默认使用其保存的原资产；
跨地形继续训练应使用显式初始化迁移，不是同配置严格 resume。

## 查看与重建

查看当前已生成资产，无需安装 MuJoCo/MJLab：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py --family boxes --level 9 --view region --skip-captures
```

窗口标题为 `pgmt_stage2_v8 Patch Inspector`，可切换 7 类地形和 L0–L9。
`Overview` 显示完整地图和围墙，`Region`/`Patch` 单独显示训练地块。
若查看旧资产，显式传入 `--asset assets/terrain/pgmt_stage2_v6` 等路径。

新版本重建分两步。先准备含 `mjlab==1.4.0`、`mujoco==3.8.0`、
`mujoco-warp==3.8.1`、NumPy、trimesh、PyYAML、rtree 的 Python 环境，
再用 Isaac Lab 环境写 USD。本机已准备好隔离的 `/tmp/pbfm-terrain-viewer-env`，
仅导出时需要；正式训练只读取生成的 USD/PLY。
以下命令要求输出目录尚无完整 `manifest.json`；构建器拒绝覆盖完整资产。

```bash
/tmp/pbfm-terrain-viewer-env/bin/python scripts/export_pgmt_pbfm_source.py \
  --config configs/terrain/pgmt_stage2_v8.yaml --output assets/terrain/pgmt_stage2_v8

OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/build_pgmt_patch_terrain.py --config configs/terrain/pgmt_stage2_v8.yaml --headless --device cpu
```

重建已有版本时应复制配置、修改版本名并使用新的输出目录，保留原资产供历史 checkpoint 使用。

## 验证记录

2026-11-11 验证结果：

- 原生 MuJoCo 与转换网格：47,616 条射线，最大误差 `2.98e-9 m`。
- Isaac Sim 6 / PhysX：49,105 条射线，包含所有训练地块、保护区、围墙、
  1,120 个候选点和 369 个接缝点；最大误差 `1.34e-6 m`，接缝误差为零。
  报告：`logs/pgmt_v8_physx_validation/collision_validation.json`。
- 真实 Isaac Lab / G1 环境 CPU 检查：10 个环境覆盖 7 类地形，L0/L5/L9
  各完成 reset 和 3 个控制步；出生间隙约 3 mm，扫描有效点各 4,410，
  七类环境数量 `[2,1,2,2,1,1,1]`；外围地形查询与无课程标签行为通过。
  报告：`logs/pgmt_v8_runtime_smoke.json`。
- 相关单元/回归测试：125 passed，1 skipped（沙箱无 CUDA，CPU 测试通过）。

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_pbfm_source.py tests/test_pgmt_pbfm_terrain.py \
  tests/test_pgmt_patch_layout.py tests/test_pgmt_terrain_query.py \
  tests/test_pgmt_curriculum.py tests/test_pgmt_contact_state.py \
  tests/test_pgmt_tracking_relaxation.py tests/test_pgmt_terminations.py

OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py --headless --validate-only --gpu 0 \
  --output logs/pgmt_v8_physx_validation
```

这些检查覆盖地形转换、运行时接线和短程物理步进，不代表策略已在新地形收敛。

中心初始化变更另行验证：76 项相关单元/回归测试通过；真实 Isaac Lab/G1 CPU 环境
覆盖七类地形的 L0/L5/L9、普通和加速重置、非连续环境索引的局部重置，共检查 138 个出生姿态。
根部 XY 与地块中心最大误差 `4.66e-10 m`，朝向随重置变化，约 3 mm 的碰撞间隙检查通过，
局部重置不改变其余环境的参考变换。报告：`logs/pgmt_center_reset_smoke.json`。
