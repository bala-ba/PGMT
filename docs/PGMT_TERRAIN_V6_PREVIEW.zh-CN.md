# PGMT v6：重复小块地形预览

2026-10-11 更新：五类地形定义全部直接调用 Isaac Lab 原生接口。现已作为可选地形接入盲走训练，支持小块内随机出生，见 [训练测试记录](PGMT_V6_BLIND_TEST.zh-CN.md)。v5 配置、生成器、查看器和资产保持原样；训练入口默认仍为 v5，通过 `--terrain_asset assets/terrain/pgmt_stage2_v6` 选择新版。

## 原生接口

| v6 样式 | 原生配置类 |
| --- | --- |
| flat | `MeshPlaneTerrainCfg` |
| stairs | `MeshInvertedPyramidStairsTerrainCfg` |
| slope | `HfInvertedPyramidSlopedTerrainCfg` |
| boxes | `MeshRandomGridTerrainCfg` |
| rough | `HfRandomUniformTerrainCfg` |

`configs/terrain/pgmt_stage2_v6.yaml` 中每类均显式声明 `class_name`。`native_config()` 构建原生配置，`call_native()` 调用其 `cfg.function(difficulty, cfg)`；不再调用旧版自定义 `generate_tile()`。外围四条平面也使用 `MeshPlaneTerrainCfg`。

保留的适配负责难度参数换算、独立随机种子、区域排列、网格合并和导出。rough 仅清除原生陡面修正产生的零面积三角形，不平滑或裁剪高度；boxes 在 `platform_width=0` 时清除原生函数返回的零面积平台三角形。slope 和无平台 boxes 的 origin 元数据查询实际表面，不重写原生网格生成规则。

## 布局与参数

- 5 种样式：`flat / stairs / slope / boxes / rough`，10 级难度。
- 每个“样式 × 难度”区域为 **24 × 24 m**，由 **4 × 4 个 6 × 6 m 小块**组成，共 **50 个区域、800 个小块**。
- stairs 和 slope 均复用 Isaac Lab 原生 inverted pyramid 生成器：中心为负高度低平台，向四周上行。没有覆盖凹地的整图零高度平面。
- 同一区域共享 difficulty；同一级的五种样式也共享 difficulty。固定种子 `1731`，各级取 `d=(level+u)/10`。实际 L0–L9 的 d 记录于资产 manifest；另检查 d=0、1 两端点。
- stairs：高度 `0.05–0.30 m`，**所有难度踏面深度固定 `0.32 m`**，暂时关闭深度变化，外围边带 `1 m`。配置保留 range 形式 `[0.32, 0.32]`，便于以后恢复变化。
- slope：输入角度 `0–30°`，外围边带 `0.25 m`；中央平台参数 `platform_width` 从 `2 m` 缩至 **`0.5 m`**，只覆盖 slope 的平台宽度，不影响其他样式。保留原生高度场离散化，因此实际平底形状和面积不等同于精确的 `0.5 × 0.5 m` 正方形。
- boxes：原生 `grid_height_range=[0.05,0.40]` 表示**高度扰动幅度 H** 随难度增加，各网格顶面在 `[-H,+H]` 内随机采样；不再是全部为正的 `0.05–0.40 m` 箱体。`grid_width=0.70 m`、`holes=false`、**`platform_width=0`，去除中央平台**；中心同样由随机网格组成。
- rough：输入噪声幅度 `±0.01–0.08 m`，网格间距 `0.15 m`，保留原生插值及陡面修正。插值输出可能超过输入噪声幅度。
- boxes / rough 每个小块有独立固定随机种子；stairs / slope 的同级小块几何相同。
- 有效地图 XY 为 `[-120,120] × [-60,60] m`；外围另保留 `20 m` 展示边带。

原生 boxes 在每个 6 m 小块内生成 8×8 个网格，四周各留 `0.2 m` 的零高度边带，相邻小块连接带总宽 `0.4 m`。连接带本身仍为 `z=0`，箱面现在可以高于或低于它；没有额外的 `z=0` 底平面覆盖负高度网格。

用户已确认 `0.4 m` 指两个小块之间的**连接总宽**，因此继续保持每侧 `0.2 m`。配置 `connection_width=0.4` 是核验约束；原生接口按 `6-floor(6/0.7)*0.7` 得到这一宽度，没有另建连接平面。

## 查看

在 PMT 根目录运行：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py
```

默认显示 `stairs / L5` 的完整 16 小块区域。窗口 `PGMT v6 Patch Inspector` 支持：

| 控件 | 用途 |
| --- | --- |
| Terrain / Level | 切换地形样式和 L0–L9 难度 |
| Patch | 选择 P00–P33；编号为 `row*4+col` |
| Overview | 显示整个地图 |
| Region (16) | 显示所选样式和难度的 16 个小块 |
| Patch / Close | 显示单个小块 / 拉近检查 |
| Show candidate XY / yaw | 黄色点表示随机候选位置，线表示朝向 |

例如直接查看最高难度箱体：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py --family boxes --level 9 --patch 5 --view patch
```

候选点在距小块外边界 `0.25 m` 的范围内均匀采样 XY，yaw 在 `[-π,π]` 采样，Z 查询实际表面。每块展示 16 点，**不是机器人有效初始姿态**：预览标记本身不执行全身碰撞或动作兼容性过滤。训练 reset 已另外接入全身碰撞、兼容表和地图边界检查；随机化出生位置与朝向，动作起始帧仍固定为第 0 帧，脚部有效支撑起始限制保持关闭。

直接查看更新后的 slope：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py --family slope --level 5
```

平台缩小后，原生 height-field origin 仍使用固定 `2 × 2 m` 范围内的最高点，会落到坑底上方。v6 slope 现将 origin Z 设为实际中心表面高度，原生 origin 数值另保存在 manifest 的 `native_origin_z_m` 中供追溯。

boxes 去平台后也不能继续使用原生返回的 `origin.z=+H`。预览 origin 改为查询中心附近网格内部点的实际表面；XY 相对几何中心偏移 `(0.031,0.019) m`，避开四个箱面相交位置的射线歧义。原生 origin Z 同样保留在元数据中；训练另按随机 XY 与全身碰撞包络查询实际放置高度。

十级 slope 的实测平底面积均下降；例如 L5 从约 `6.60 m²` 降至 `0.26 m²`，L9 从约 `6.28 m²` 降至 `0.26 m²`。最低难度的高度量化会使平底更大，L0 约 `0.81 m²`。对比记录为 `logs/pgmt_v6_parameter_change_20261011.json`。

## 已完成检查

构建日志：`logs/pgmt_terrain_build_v6.log`。实际 Isaac / PhysX 检查结果：`logs/pgmt_terrain_preview_v6/collision_validation.json`。

| 检查 | 结果 |
| --- | --- |
| 50 区域、800 小块的几何生成 | 通过 |
| 5 样式的 d=0、1 参数端点 | 通过 |
| 倒金字塔四方向上行、中心负高度 | 通过 |
| 160 个 boxes 小块具有正负顶面、相邻网格无额外间隙 | 通过 |
| boxes 中央平台关闭、连接总宽 0.4 m | 中央面尺寸以及四侧边带内外采样点检查通过 |
| rough 有正负高度变化 | 通过 |
| mesh 与 PhysX 高度查询 | 60,640 个几何采样点通过 |
| 随机候选点表面 Z | 12,800 个采样点通过 |
| 小块接缝及两侧高度 | 4,620 个采样点通过，误差为 0 |
| 总计 78,060 次查询的最大高度误差 | 约 `2.472e-6 m` |
| v5 六个关键文件的 SHA-256 | 与修改前相同 |
| v6 的 800 个 PLY 文件校验 | 通过 |
| 800 个小块的原生接口元数据 | 与五类配置一致 |
| flat / stairs / slope / rough 与去平台前预览的网格 SHA-256 | 完全相同 |

以上是预览阶段的采样点一致性验证。后续随机机器人 reset 与实际盲走训练启动检查单独记录于 [训练测试记录](PGMT_V6_BLIND_TEST.zh-CN.md)。

构建过程中修正了两处预览问题：原 24 m 地形斜坡检查点在 6 m 小块中会落到中央平台，现改为检查实际外侧斜面；USD 显隐分组补齐 Xform 类型，使区域和单块切换生效。修改仅位于新增 v6 文件中。

原生接口切换记录：`logs/pgmt_v6_native_interfaces_20261011.json`。boxes 全部网格中心、首行/首列接缝以及四侧连接边带的采样参与 mesh / PhysX 对照，负高度点也参与检查，避免漏掉覆盖平面。去平台记录为 `logs/pgmt_v6_boxes_no_platform_20261011.json`。

截图保存于 `logs/pgmt_terrain_preview_v6/`：全图 `overview.png`、五种地形的 `*_L5_region.png`、楼梯/斜坡/箱体/rough 的 `*_L9_patch.png` 及 `candidate_xy_yaw.png`。

## 文件与重建

- 配置：`configs/terrain/pgmt_stage2_v6.yaml`
- 小块生成和几何验证：`pmt_tasks/pgmt_patch_terrain.py`
- 资产构建：`scripts/build_pgmt_patch_terrain.py`
- 原生 Isaac 可视化与 PhysX 检查：`scripts/view_pgmt_patch_terrain.py`
- 资产及元数据：`assets/terrain/pgmt_stage2_v6/`
- 平台/踏深调整前的 v6 资产与源文件快照：`assets/terrain/pgmt_stage2_v6_before_20261011/`
- 平台/踏深调整前的截图与碰撞检查：`logs/pgmt_terrain_preview_v6_before_20261011/`
- 切换原生接口前的 v6 资产与源文件快照：`assets/terrain/pgmt_stage2_v6_before_native_20261011/`
- 切换原生接口前的截图与碰撞检查：`logs/pgmt_terrain_preview_v6_before_native_20261011/`
- 去除 boxes 平台前的资产与源文件快照：`assets/terrain/pgmt_stage2_v6_before_boxes_no_platform_20261011/`
- 去除 boxes 平台前的截图与碰撞检查：`logs/pgmt_terrain_preview_v6_before_boxes_no_platform_20261011/`

为保护已有资产，构建器拒绝覆盖已经生成 manifest 的目录。调整参数后可输出至新目录：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/build_pgmt_patch_terrain.py --headless --device cpu \
  --output assets/terrain/pgmt_stage2_v6_review2
```

旧地形仍可直接打开：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_terrain.py --asset assets/terrain/pgmt_stage2_v5
```

v6 importer/区域归属、小块内有效随机 XY/yaw reset、五类兼容表映射和跨块查询已接入盲走训练。小块边界本身不触发 done；出生课程等级和 episode 参考变换按原约定保持。全库随机出生、原生扫描及 v6 完整续训恢复现已完成专项仿真验收，见 [v6 P1 补充验收](PGMT_V6_P1_ACCEPTANCE.zh-CN.md)。盲走入口仍关闭扫描输入；后续 P2 再接入策略融合。
