# PGMT 地形构建与 Isaac 检查

当前版本为 **pgmt_stage2_v5**：楼梯踏面深度更新为 **0.25–0.35 m**，普通楼梯和倒金字塔楼梯共用此范围。五类原有地形加上倒金字塔上楼梯、上坡，共 **七种样式 × 十级，70 块 24×24 m 地形**。地形资产、几何检查及 Isaac PhysX 验证已完成；stage 2 的动作采样、重置、感知网络和训练尚未接入。

| 类型 | 范围与实现 |
| --- | --- |
| flat | 平地，各级保持一致 |
| slopes | `HfPyramidSlopedTerrainCfg`：中心高、四周低，从中心向外下坡；坡角 0°–30°，按角度线性递增后转换为 `tan(theta)` |
| stairs | `MeshPyramidStairsTerrainCfg`：中心高、四周低，从中心向外下楼；阶高 0.05–0.30 m，踏面深 0.35→0.25 m |
| boxes | 紧密拼接的 0.70 m 方格，无间隙；每块 34×34 个箱格，箱顶采样 `[0.05,h(d)]`，`h(d)=0.05+0.35d` m；中心 2 m 平台高度为 `h(d)` |
| random rough | PGMT 原生 `HfRandomUniformTerrainCfg`；输入噪声半幅 0.01–0.08 m，噪声步长 0.01 m、粗采样 0.30 m、网格 0.15 m、高度量化 0.005 m、边框配置 0.25 m、陡面阈值 0.75 |
| up_stairs | **`MeshInvertedPyramidStairsTerrainCfg`**：中心低、四周高，从中心向外上楼；与 stairs 共用阶高、踏面深度、边框及中心平台配置 |
| up_slope | **`HfInvertedPyramidSlopedTerrainCfg`**：中心低、四周高，从中心向外上坡；与 slopes 共用坡角、分辨率、边框及中心平台配置 |

v1 把 boxes 做成了间隔 0.20 m 的独立障碍，并对 rough 使用更密网格、不同噪声步长和插值限幅，改变了参考项目的样式；v2 修正这些差异。boxes 保留用户指定的**正高度**语义，没有照搬 PGMT 的 `[-h,+h]` 箱顶分布。rough 则恢复原生三次插值、不限幅：**1–8 cm 是输入噪声半幅，并非最终表面高度硬上限**；v2 的 L9 实际范围约为 −0.130～+0.125 m，参考缓存同级约为 −0.125～+0.120 m。随机纹理采用独立固定种子，样式和分布对齐，不要求逐顶点相同。

第 L 级在生成时采样 `d∈[L/10,(L+1)/10)`，实际参数在检查面板显示。最大设计尺寸由独立 `d=1` 检查覆盖，不保证 L9 随机网格刚好达到上限。布局沿 X 增加等级，沿 Y 按上表顺序排列七种样式；四种正/倒金字塔地形的中心平台配置宽度均为 2 m，rough 的中心仍保留起伏。金字塔地形会累计多级高度，单级阶高与整个地形总高度是不同量。

踏面深度映射为 `w(d)=0.35−0.10d` m：最低难度 0.35 m、最高难度 0.25 m；阶高仍为 `h(d)=0.05+0.25d` m。

v4 按用户澄清，将 v3 误做的直线楼梯/斜坡替换为 Isaac Lab 原生倒金字塔。生成器 origin 位于中央低平台，Z 坐标为负值；四周外缘回到 Z=0。同一 difficulty 下，倒金字塔与对应正金字塔的可行走表面满足 `z_inverted(x,y) = -z_pyramid(x,y)`。从中心可向四周上行，无固定 +X 前进限制。训练重置应使用真实 origin 高度，不能把负高度截成 0；高度扫描也须覆盖中央凹地。

这两种样式是用户增加的地形扩展。manifest 的 `family_group` 将 `up_stairs → stairs`、`up_slope → slopes` 映射回论文五类语义，供动作兼容和奖励容差使用。七列均匀排布用于预览；训练采样比例需单独配置，不能直接把七列索引套入原有五类规则。

## 查看

在项目根目录运行；自动选择连接桌面的 GPU，也可用 `--gpu` 指定：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  /home/lzp/miniconda3/envs/isaacsim/bin/python scripts/view_pgmt_terrain.py
```

Isaac 的 **PGMT Terrain Inspector** 面板提供 Terrain、Level 选择，以及 **Overview / Whole tile / Close up** 按钮；默认打开 **up_stairs L5** 倒金字塔楼梯全景，选择 **up_slope** 可查看倒金字塔坡面。面板显示原生生成器类名。选定地形时暂时隐藏其他 tile，总览恢复显示全部；Close up 查看中心低平台附近。可用 `--family up_slope --level 9 --view tile` 指定初始画面。

资产：[terrain.usdc](../assets/terrain/pgmt_stage2_v5/terrain.usdc)。场景内为同一份可渲染、可碰撞的三角网格；没有训练进程或策略在运行。窗口关闭即可退出预览。旧版资产保留归档，v3 的直线地形已停用。

本次通过 headless 模式完成 v5 碰撞验证。上述查看命令默认加载 v5；启动后脚本会先运行 PhysX 校验，再显示检查面板。

## 验证与重建

- 70 块地形及 14 个端点几何检查通过：实测坡度、阶高与踏面边界、箱格顶面/接缝、rough 起伏，以及倒金字塔从中心向四个方向升高。
- 20 块倒金字塔地形逐一与相同 difficulty 的正金字塔进行表面镜像对照，最大高度误差 **1.78e−15 m**；起始 origin 在中央低平台。
- 20 块楼梯使用新踏面深度，独立检查 0.35 m / 0.25 m 端点；其余 50 块网格和 difficulty 与 v4 完全一致，见 [深度范围验证](../logs/pgmt_terrain_preview_v5/depth_range_validation.json)。v2 的 [PGMT 样式对照报告](../logs/pgmt_terrain_preview_v2/reference_style_validation.json) 仍适用于 boxes / rough。
- v5 的 **720 个 PhysX 查询全部通过**，最大高度误差 **2.18e−6 m**，见 [碰撞报告](../logs/pgmt_terrain_preview_v5/collision_validation.json)。
- [资产 manifest](../assets/terrain/pgmt_stage2_v5/manifest.json) 保存 seed、difficulty、几何统计、探针、源码/配置/网格哈希，以及 `shape`、`inverted`、中心 origin 和镜像对照结果。

```bash
# 已完成的输出目录拒绝覆盖；更改参数后指定新目录/版本。
OMNI_KIT_ACCEPT_EULA=YES \
  /home/lzp/miniconda3/envs/pmt/bin/python scripts/build_pgmt_terrain.py \
  --headless --device cpu --output assets/terrain/pgmt_stage2_v6
```

配置入口：[pgmt_stage2.yaml](../configs/terrain/pgmt_stage2.yaml)；生成实现：[pgmt_terrain.py](../pmt_tasks/pgmt_terrain.py)。训练侧可在 AppLauncher 之后调用 `make_generator_cfg()` 复用相同生成函数；生成的缓存包含源码指纹，参数或几何实现变化会生成不同缓存。重建版本须同步更新 YAML 的 `version`。
