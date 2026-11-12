# PGMT P2：感知接入实施计划

更新：2026-10-13。状态：**P2.1–P2.4 已完成，P2.5 的 160 环境集成验收通过**：指定 Stage1 权重迁移、25 次 GUI PPO、25→30→再次加载 30、三组成本对照、321 项 CPU/CUDA 回归通过。1024 环境实际尝试受 PhysX 初始化显存限制，未通过扩容；失败保护已补齐。详见 [P2.4/P2.5 验收报告](PGMT_P245_ACCEPTANCE.zh-CN.md) 与 [机器报告](../logs/pgmt_p245/report.json)。P2 保留原三头奖励，P3/P4 尚未完成。

2026-10-12 追加要求：**高度符号、无效高度、归一化位置和增强采样时机全部对齐原 PMT 视觉分支**。当前 version=3：高度为 sensor−ground；无效世界命中 Z 填 30 m；观测裁剪 ±3 m 并保留米制，除以 3 的归一化仅在网络输入层执行；每次执行观测函数重新采样 10% 丢点和 ±0.06 m 均匀噪声。单次函数生成地图/掩膜，wrapper 无随机拆分，取消按控制步/局部 reset 的增强缓存。扫描仍为 2×2 m、21×21，并保留有效地图范围掩膜。version=1/2 为历史契约；最新复验见 [P2.1 验收](PGMT_P21_ACCEPTANCE.zh-CN.md)。

**P2 的目标：让已验收的高度扫描经过可学习的 Terrain-Glimpse 与 IFM，实际参与 actor/critic，并完成迁移、短训和续训验证。** 推荐顺序为「固定输入契约 → 独立验证 glimpse → 接入 IFM → 权重迁移与训练入口 → Isaac 验收」。

## 1. 范围与已有基础

| 项目 | 当前状态 / P2 处理 |
| --- | --- |
| 地形、参考与采样 | 直接复用 v6 五类十级、800 小块、随机出生 XY/yaw、固定动作第 0 帧、CSV 兼容过滤和动作级 adaptive sampling；P1 已验收 |
| 地形观测 | `PGMTPlacementEnvCfg` 已提供 torso 中心、yaw 对齐的 2×2 m、21×21 扫描及有效掩码，P1 已覆盖全地图；P1 盲走配置则主动移除了扫描器 |
| 既有网络 | 已有 history encoder、8 帧未来 command、带 RoPE 的 IFM、三头 critic；Terrain-Glimpse 已接入 `_encode()` 的 13 个 K/V，观测分组、整网诊断及元数据已完成；原生 PPO 更新、真实地图 CPU/CUDA 融合验收与整网成本记录均完成 |
| 可复用实现 | `../PGMT/pgmt/modules.py` 已有可微裁剪与融合原型。移植纯张量逻辑并适配 PMT；其 border padding、中心范围及 checkpoint 行为不能直接当作本项目验收结论 |
| P2 / P3 边界 | P2 短训保留 `[upper, lower, auxiliary]` 三组奖励和三头 critic。**第四头与六项 terrain-contact 奖励在 P3 同时启用**；P2 预留按语义迁移接口，不训练一个没有对应奖励的第四头 |

P0 中尚未落地的独立任务配置、迁移入口和 schema 校验一并纳入 P2；四头映射移至 P3。P2 的三头配置明确标为 `perception_only`，不是完整 Stage 2。P3 还负责地形跟踪容差、终止延迟与恢复补课；P4 承担正式训练和性能评测。

## 2. 实施任务与顺序

| 顺序 | 任务 | 主要交付 | 通过条件 |
| --- | --- | --- | --- |
| P2.1 ✅ | 固定感知契约与配置 | 独立 `PGMTPerceptionEnvCfg`、version=3 PMT 语义；每次观测函数采样一次原子地图/掩膜；网络侧 `PGMTTerrainInput` 预处理层 | 同一物理输入与 RNG 下原 PMT 精确对照、同一步重复调用重采样、局部 reset 后重新观测、原生 rollout/minibatch 及网络确定性归一化均通过；147 项回归、800 小块/10,786 条 PhysX 射线复验通过 |
| P2.2 ✅ | Terrain-Glimpse | 输入、selector、带掩膜 5×5 裁剪、共享 patch encoder、调试输出及独立验收完成 | 157 项回归无跳过；CPU/CUDA 与 NumPy 对照、有限差分及完整 token 梯度通过；中心分布与 batch 1/160/1024/8192 耗时/显存已记录 |
| P2.3 ✅ | 扩展 IFM | 13 个 K/V、逐环境掩膜、共享 actor/critic、观测分组、元数据及诊断完成；原生 PPO 和真实地图融合验收通过，285 项相关回归无跳过，整网 batch 160/1024 成本已记录，详见第 3.7–3.12 节 | 关闭感知逐值回到旧行为，全无效回退误差 ≤5.96e-8；有效地图影响输出，PPO 梯度到达感知模块；CPU/CUDA 动作/value/intent 最大误差 <1.5e-7 |
| P2.4 ✅ | 迁移、保存与训练入口 | 独立任务/网络配置、显式初始化与续训模式、完整 checkpoint 元数据 | 指定 Stage 1 权重逐项迁移；P2 同配置二次保存/加载后模型、采样器、课程及训练状态正确恢复 |
| P2.5 ✅（160 环境） | Isaac 可视化与短训 | glimpse 覆盖图、各项诊断、短训报告和复现命令 | 五类地形实际 PPO 更新、reset/跨 tile/地图边缘检查、保存恢复通过；报告显存和吞吐；1024 扩容因 PhysX 显存不足未通过，详见 §7 |

先完成 P2.1–P2.3 的纯张量验证，再接仿真。P2.4 可复用现有入口的配置和 GUI 控件；P2.5 通过后进入 P3。

## 3. 网络与数据契约

```text
当前 proprio [N,96] + history [N,10,96]
    → 既有 history encoder → history token [N,1,128]

米制地图 [N,21,21] + 掩膜 [N,21,21]
    → 网络 PGMTTerrainInput：clip(-3,3)/3，高度与 mask 分别处理
归一化地图 + 掩膜 + history token + future commands [N,8,61]
    → selector → 4 个中心 [N,4,2]
    → bilinear crop → patches [N,4,5,5]
    → patch MLP（25 个高度 + 25 个有效性值 + 2 个中心坐标）→ terrain tokens [N,4,128]

Q = history token
K/V = history token + 8 个 command token + 4 个 terrain token
    → IFM → intent [N,128]
    → actor [N,29] / critic [N,3]（P2）；critic [N,4]（P3）
```

论文明确使用 4 个 5×5 glimpse、可微双线性裁剪、状态和未来动作条件化的 selector，以及共享 IFM。以下宽度、坐标、边界、掩码和噪声参数是 **PMT 工程选择**，统一登记在配置与 checkpoint 中。

| 约定 | 首版建议 |
| --- | --- |
| 高度与轴序 | `map[row=y,col=x]`，两轴为 torso yaw 坐标的 `[-1,+1] m`；策略高度为 `sensor_z - ground_z - offset`，offset=0，地面低于 torso 为正。20 m 起射偏移不参与高度零点。干净几何 `terrain_scan()` 的历史 ground−torso 接口保持不变 |
| 地图预处理 | ObservationManager 单次生成 `pgmt_terrain[N,882]`，wrapper 无随机拆为 `terrain_map` / `terrain_valid`，不拼入 proprio/privileged。无效世界 Z 填 30 m，sensor Z 非有限回退 0.78 m；相对高度加噪后裁剪 ±3 m，观测不归一化。网络 `PGMTTerrainInput` 按原 PMT 再裁剪并除以 3；无逐格运行均值归一化。无效观测通常为 −3 m，网络输入通常为 −1，并保留 mask=0 |
| selector | 输入 `441+441+128+8×61=1498`，MLP `1498→256→128→8`；显式读取网络归一化高度和丢点后的掩膜，以区分真实测量与无效填充。增加 mask 输入是适配 PMT 数据增强的工程选择 |
| 中心与采样 | `center_xy = 0.8 × tanh(raw)`，使 5×5 采样网格完整落在扫描方形内；offset 为 `[-0.2,-0.1,0,0.1,0.2] m`，五点跨度 0.4 m；`align_corners=True`，显式做米制到归一化 grid 坐标转换 |
| 有效性 | 同一网格采样 `map×valid` 与 `valid`，用有效权重归一化插值，防止无效填充值污染高度；权重 `<0.5` 的采样点填 −1 并标无效。首版 patch 至少 13/25 点有效才进入 K/V，其余屏蔽；全无效回退 history/command。采样 padding 的权重为 0，不做 border replication。覆盖率阈值需在 P2.2 验证并写入配置/checkpoint |
| patch encoder | 共享 MLP `52→128→128`，输入高度、逐点有效性和中心 `(x,y)`；中心仍为 torso yaw 坐标、单位 m。新增 mask 和覆盖率规则属于工程选择，不作为论文公开超参数 |
| RoPE | 保留 history/query 位置 0、command offsets `[0,1,3,7,15,31,63,127]`；terrain 位置设 0，表示当前地图。该模态位置规则沿用本地原型，属于工程选择 |
| 增强 | 默认对齐原 PMT `VisionCfg`：每点独立丢失概率 0.1，均匀噪声 ±0.06 m，观测端裁剪 ±3 m；每次执行观测函数都重新采样，不按物理步缓存。只修改策略副本，sensor hits/干净几何不变。`augmentation_enabled=False` 仅关闭随机丢点/噪声，仍保留 PMT 符号、填充和裁剪；原始几何直接读 `terrain_scan()` |

不再要求 patch 所有点有效：独立丢点率 10% 时，即使网格对齐，一个 5×5 patch 的完整保留概率也仅 `0.9^25≈7.18%`。P2.2 第 3–4 项已验证随机丢点下的有效 patch 比例、缺测点不污染插值、全无效时有限 token 与全 false 掩膜，以及 token 损失经裁剪高度到 selector 的梯度；P2.3 第 2 部分进一步验证了 `_encode()` 中的逐环境屏蔽、9-key 回退和 intent 经 crop 到 selector 的梯度。

两个实现要点：

- **PPO 必须重算可微感知网络。** rollout 存地图和掩码，更新时从 minibatch 重新计算 selector/patch/IFM；不能把 rollout 时 detach 的中心或 token 当作唯一训练输入。
- **噪声和丢点在每次观测函数执行时采样。** 同一调用原子地产生地图与掩膜；同一物理步再次调用会为全部环境重新采样，局部 reset 后下一次全量观测也是如此。actor/critic 和 PPO 必须使用已保存的同一观测，不重新请求环境观测；网络归一化是确定性计算。version=1/2 不允许静默按 version=3 续训。

`PGMTTerrainGlimpse` 已串联输入、selector、crop 和共享 patch encoder。`PGMTTerrainGlimpseInput` 每次完整 forward 仅调用一次已有 `PGMTTerrainInput`；后续模块消费同一份归一化数据，不再次归一化或增强。P2.3 已接通 terrain K/V；默认 Stage-1 实例仍不读取地图，P2.4 的正式 `pgmt_stage2` 配置与 `train_pgmt_stage2.sh` 入口默认开启感知。

### 3.1 P2.2 实施步骤（第 1–5 项完成）

**目标：实现独立的 Terrain-Glimpse 模块，根据历史状态和未来动作，从地图中选择 4 个局部区域，输出可微的地形 token 及有效性标记。** 对应论文式（7）—（8）；本阶段完成局部编码的数值、几何和梯度验收，P2.3 再接入 IFM 并验证 PPO 梯度。4 个中心由训练学习，不预先绑定脚的位置，也不需要落脚点标注。

1. **✅ 固定模块接口，复用感知预处理。** 已新增 [pgmt_terrain_glimpse.py](../motion_tracking_rl/networks/pgmt_terrain_glimpse.py) 的 `PGMTTerrainGlimpseInput`。输入为米制 `terrain_map[N,21,21]`、`terrain_valid[N,21,21]`、`history_token[N,1,128]` 和 `commands[N,8,61]`。地图调用已有 [PGMTTerrainInput](../motion_tracking_rl/networks/pgmt_terrain_input.py)；history token 和 commands 原样传递，在后续集成时直接取自现有 [_encode()](../motion_tracking_rl/networks/pgmt_actor_critic.py) 的 history attention 输出及 command normalizer 输出，不另建归一化器。高度符号、无效世界 Z 填充和增强采样沿用 P2.1 version=3，模块内部不生成随机噪声。

2. **✅ 实现位置选择器。** 已新增 `PGMTTerrainPositionSelector` 和集中配置 `PGMTTerrainSelectorCfg`。按第 3 节契约拼接归一化高度、掩膜、历史 token 和未来命令，默认使用 `1498→256→128→8`、ELU 的 MLP，输出 `center_xy[N,4,2] = 0.8×tanh(raw)`。中心为 torso yaw 坐标，单位 m；约束在 `[-0.8,0.8]`，保证周围 ±0.2 m 的窗口位于扫描方形内。实际地图边缘的缺测由 crop 掩膜处理。网络宽度、激活和中心范围属于工程选择，详见第 3.3 节。

3. **✅ 实现带掩膜的可微裁剪。** 已新增 `PGMTMaskedTerrainCrop` 和 `PGMTTerrainCropCfg`。每个中心采样 5×5 点，间隔 0.1 m、两端跨度 0.4 m。显式转换米制坐标到 `grid_sample` 坐标，采用双线性插值、`align_corners=True`、`row=y,col=x`。同一网格采样有效高度与掩膜，以有效权重为分母并加安全下限；权重不足 0.5 的点输出高度 −1、点掩膜 false。超出扫描方形的点权重为 0，不复制边界高度。首版至少 13/25 点有效才保留 patch；全无效时返回有限值及全 false 的 `patch_valid`，供后续 token 使用。阈值配置化并完成丢点统计，详见第 3.4 节。

4. **✅ 实现共享 patch encoder 和调试输出。** 已新增 `PGMTTerrainPatchEncoder`、配置 `PGMTTerrainPatchEncoderCfg` 及完整入口 `PGMTTerrainGlimpse`。每个 patch 拼接 25 个高度、25 个有效标记、2 个米制中心坐标，经过同一个 `52→128→128` MLP。返回 `terrain_tokens[N,4,128]`、布尔 `token_valid[N,4]`；`return_debug=True` 时额外返回中心、高度、点掩膜、有效插值权重、覆盖率和 token 掩膜的独立 `detach().clone()` 副本。主计算路径保持梯度连通。详见第 3.5 节；无效 token 的 attention 屏蔽及全无效时仅使用 history/command 的逻辑由 P2.3 实施。

5. **✅ 完成独立验收并记录结果。** 已新增 `scripts/validate_pgmt_terrain_glimpse.py`，汇总相关测试并执行实际地图 NumPy 对照、CPU/CUDA 前向/反向、掩膜压力样例、有限差分、中心分布及批量耗时/显存检查。157 项回归无跳过，独立验收通过，详见第 3.6 节与 [P2.2 验收报告](PGMT_P22_ACCEPTANCE.zh-CN.md)。可以进入 P2.3。

P2.2 的验收清单：

| 检查 | 通过条件 |
| --- | --- |
| 几何与插值 | 常数平面、X/Y 单轴斜坡、台阶、非整数中心及边缘位置的裁剪与预期一致；覆盖 N=1 和批量，验证轴序及实际 yaw 坐标映射 |
| 掩膜与缺测 | 无效填充值不污染有效高度；覆盖 10% 随机丢点、局部缺测和整图无效；输出无 NaN/Inf，并报告各类输入的有效 patch 比例 |
| 梯度 | 有效非平坦图上 selector、patch encoder 梯度有限且非零；用有限差分单独验证裁剪高度对中心的导数，排除仅通过中心坐标拼接产生的旁路梯度。检查点避开插值网格结点和有效性阈值边界，不要求常数图或全无效图的选点梯度非零 |
| 确定性 | 同一份已经采样保存的观测重复前向，结果一致；模块内部不重新采样增强、不修改原始地图和掩膜 |
| 实际地形 | 使用 `logs/pgmt_p21_acceptance/20261012_pmt_alignment_v3_r1/terrain_map_L{0,5,9}.npz` 及 `pmt_augmented_maps_L9.npz`，检查五类地形的裁剪位置、高度、覆盖率和无效区域；不能混用旧 version=1/2 的符号或归一化数据 |
| 批量与设备 | CPU/CUDA 前向、反向正常；记录批量计算耗时和显存，并检查中心重叠、边界饱和情况。4 个 crop 网格可在同一地图上批量采样，避免为每个 patch 复制整张地图 |

可参考 [本地原型](../../PGMT/pgmt/modules.py) 的 `crop_patches()`，但须替换其 border replication，并补充掩膜和有效权重插值。P2.2 的交付物为独立模块、集中配置、测试及数值/梯度验收记录；checkpoint 元数据保存由 P2.4 接入。完整 IFM 融合、迁移、Isaac PPO 短训仍按 P2.3–P2.5 执行，独立模块通过不代表策略已经学会选择落脚点。

### 3.2 P2.2 第 1 项实施记录（2026-10-12）

当前入口为 `PGMTTerrainGlimpseInput.forward(terrain_map, terrain_valid, history_token, commands)`，返回具名元组 `PGMTTerrainGlimpseInputs`，包含以下字段：

| 字段 | 当前输出契约 |
| --- | --- |
| `terrain_map` | `[N,21,21]`，网络侧 `clip(-3,3)/3` 后的无量纲高度，保留原轴序和无效填充值；不是米制地图 |
| `terrain_valid` | `[N,21,21]`，按已有预处理的 `>0.5` 规则转为 0/1，并转换为地图 dtype；没有额外丢点 |
| `history_token` | `[N,1,128]`，调用者 tensor 原样返回，保留 autograd 连接 |
| `commands` | `[N,8,61]`，调用者已经归一化的 command tensor 原样返回，不再次归一化 |

模块检查固定形状、相同 batch/device；地图、history 和 commands 必须使用相同浮点 dtype，mask 接受布尔或浮点类型。没有隐式设备搬运，没有新参数、运行统计或 RNG 调用。输入应是 P2.1 生成的有限值观测；模块不重新修复传感器值。返回 tensor 按只读使用，特别是原样传递的 history/commands。该输入模块本身不输出采样中心或 terrain token；第 2 项 selector 消费其结果，见第 3.3 节。

新增 [test_pgmt_terrain_glimpse.py](../tests/test_pgmt_terrain_glimpse.py)，验证米制到网络归一化、非连续地图轴序、PMT 无效填充值保留、实际增强地图的原子打包/拆分、重复前向及 train/eval 无随机变化、输入不变、输入梯度保留、形状/batch/dtype/device 错配拒绝。通过实际 Stage-1 forward hook 获取 history/已归一化 commands，验证原样传递且旧策略输出和 state_dict 不受影响。

第 1 项实施时输入接口与相关回归 **74 passed，1 skipped**。当时 CUDA 在测试进程中不可用，因此跳过 1 项 CUDA 输入前向/反向检查；CPU 输入梯度已验证。该历史结果不包含 selector/crop/patch encoder 梯度，也没有新增 Isaac 或 PPO 训练验收。

复现命令：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt_terrain_glimpse.py tests/test_pgmt_perception.py tests/test_pgmt_rl.py tests/test_normalization_resume.py
```

### 3.3 P2.2 第 2 项实施记录（2026-10-12）

实现位于 [pgmt_terrain_glimpse.py](../motion_tracking_rl/networks/pgmt_terrain_glimpse.py)：

| 项目 | 当前实现 |
| --- | --- |
| 入口 | `PGMTTerrainPositionSelector.forward(prepared: PGMTTerrainGlimpseInputs)`，拒绝直接传入原始观测字典；不在 selector 内调用地图预处理 |
| 拼接顺序 | 归一化地图 441、有效掩膜 441、history token 128、future commands 488，共 1498 维。地图按 row=y/col=x 展平，命令按帧/特征展平 |
| MLP | 复用 PMT `MLP`，默认 `1498→256→128→8`、ELU 隐藏层、原生 Linear 初始化，共 417,672 个可学习参数 |
| 输出 | `[N,4,2]` 的 `(x,y)` 米制中心，实际 torso yaw 坐标系；`center_limit_m×tanh(raw)`，默认范围 ±0.8 m |
| 配置 | 冻结的 `PGMTTerrainSelectorCfg(hidden_dims=(256,128), activation="elu", center_limit_m=0.8)`。宽度须为正整数，中心范围可缩小，但不得超过 0.8 m；配置属于工程选择 |
| 随机性与梯度 | 仅参数初始化使用随机数；forward 不采样、不修改输入、不缓存 detach 结果，梯度保留到 selector 参数、地图、history 和 commands |

调用方式（模块构建一次，随后每个 forward 使用当前观测）：

```python
terrain_input = PGMTTerrainGlimpseInput()
selector = PGMTTerrainPositionSelector()

# terrain_map 为米制观测；commands 已经由原有 command normalizer 处理。
prepared = terrain_input(terrain_map, terrain_valid, history_token, commands)
center_xy_m = selector(prepared)
# 后续裁剪应直接使用 prepared.terrain_map / prepared.terrain_valid。
```

测试沿用第 3.2 节命令，第 2 项实施时相关回归 **97 passed，2 skipped**。新增覆盖：四类输入顺序与条件响应、预处理仅调用一次、XY 配对、饱和时的扫描边界、全无效但有限的地图、重复前向与 batch 重排、所有 selector 层和三类连续输入的有限非零梯度、中心导数有限差分、配置/权重严格重载。两项跳过分别是输入模块和 selector 的 CUDA 前向/反向，原因是当时测试进程 CUDA 不可用。

另用 P2.1 version=3 保存的五类地形 L0/L5/L9，以及 L9 增强地图，共 20 张实际仿真采集地图离线验证。随机初始化 selector、零 history/归一化 command 条件下，全部输出有限且不越界，中心坐标范围为 `[-0.059984, 0.193918] m`；所有 selector 层和地图输入均收到有限非零梯度，重复前向逐值相同且无新增 RNG 消耗。记录含配置、种子、中心坐标与源码/输入哈希，见 [selector 第 2 项离线记录](../logs/pgmt_p22_acceptance/selector_step2_20261012_151916.json)。

第 2 项只验证位置预测与梯度连通性；全无效地图也可能产生中心坐标，并不表示这些位置有效。crop 与 terrain token 的后续实现见第 3.4–3.5 节；IFM/PPO 注入仍待实施。模块级配置/权重重载不等于 P2.4 的完整训练续训验收。

### 3.4 P2.2 第 3 项实施记录（2026-10-12）

[PGMTMaskedTerrainCrop](../motion_tracking_rl/networks/pgmt_terrain_glimpse.py) 接收同一份 `PGMTTerrainGlimpseInputs` 和 selector 输出的 `center_xy_m[N,4,2]`。裁剪不再次归一化、不采样增强，也不修改输入。4 个 5×5 网格纵向合并为 20×5，单次 `grid_sample` 同时采样高度分子与有效权重两个通道，每个环境只保留一张输入地图。

计算规则：

- 采样坐标为中心加 `[-0.2,-0.1,0,0.1,0.2] m` 偏移；按 `grid = 2×(xy−midpoint)/scan_size` 转为归一化坐标，当前 midpoint=(0,0)、scan_size=(2,2)。使用 `align_corners=True` 和零 padding，行对应 y、列对应 x。
- 源有效性为输入掩膜有效且高度有限。用 `where(valid,height,0)` 构造分子输入，避免 `NaN×0` 仍为 NaN；与二值有效掩膜一起双线性采样。高度为 `numerator / max(valid_weight,1e-6)`。
- 显式将扫描方形外或非有限坐标标无效并令权重为 0，避免扫描边缘外仍因部分插值权重而被判有效；不通过坐标 clamp 或 border replication 伪造地面。
- `valid_weight >= 0.5` 的点保留插值高度，否则输出 −1。至少 13/25 点有效则 `patch_valid=True`。判定阈值是离散边界；有效插值区域内保留对地图和中心的梯度，无效点高度梯度为 0。

返回 `PGMTTerrainPatches`：

| 字段 | 形状与语义 |
| --- | --- |
| `height` | `[N,4,5,5]`，**网络归一化高度**，无效点填 −1；不是米制高度 |
| `point_valid` | `[N,4,5,5]`，逐点 bool 掩膜，后续 patch encoder 转成高度 dtype 后拼接 |
| `valid_weight` | `[N,4,5,5]`，有效邻点贡献的双线性权重，用于诊断；扫描方形外为 0 |
| `coverage` | `[N,4]`，有效采样点数量 / 25，不是平均插值权重 |
| `patch_valid` | `[N,4]`，patch 级 bool 掩膜，供后续 terrain token 和 IFM 使用 |

阈值集中在 `PGMTTerrainCropCfg(min_valid_weight=0.5, min_valid_points=13, weight_epsilon=1e-6)`，含合法范围校验。几何尺寸固定沿用当前 21×21、2×2 m、4 个 5×5 patch 契约。

```python
crop = PGMTMaskedTerrainCrop()  # 与网络一起构建并搬到相同设备。
patches = crop(prepared, center_xy_m)
```

第 3 项实施时验证沿用第 3.2 节命令：**137 passed，3 skipped**。覆盖常数平面、X/Y 斜坡、4 个 yaw 下的轴序、台阶与非整数中心、缺测邻点的有效权重归一化、无效填充值/NaN 不污染裁剪、12/25 与 13/25 的边界、扫描边界外部分插值、全无效与非有限坐标的有限零梯度。有限差分检查包含缺测邻点时分母对中心的导数；仅以 crop 高度构造损失即可回传 selector 所有层，未借助中心坐标拼接或额外 selector loss。3 项 CUDA 前向/反向检查因当时进程 CUDA 不可用而跳过。

实际地形离线对照使用 P2.1 version=3 的 20 张地图（五类 L0/L5/L9 及 L9 增强图），分别取 selector 中心与扫描四角中心，共 4,000 个采样点，与独立 NumPy 逐点双线性计算比较：逐点及 patch 掩膜全部一致，最大归一化高度误差 `1.87933e-7`，最大权重误差 `1.47157e-6`。另在 10% 独立丢点下分别检查 4,096 个网格对齐和 4,096 个非整数中心 patch；两组保留率均为 100%，有效点比例约 90.09%，严格 25/25 有效仅保留约 7.69%。这是固定种子样本统计，不是保留率保证。

记录含输入/源码哈希与每类结果，见 [crop 第 3 项离线记录](../logs/pgmt_p22_acceptance/crop_step3_20261012_154859.json)。该步骤未启动 Isaac 或 PPO；patch encoder 后续实现见第 3.5 节。crop 对非有限值的局部处理不替代上游版本/单位契约，也不能保证错误输入下的 selector 正常。

### 3.5 P2.2 第 4 项实施记录（2026-10-12）

[pgmt_terrain_glimpse.py](../motion_tracking_rl/networks/pgmt_terrain_glimpse.py) 新增共享 `PGMTTerrainPatchEncoder` 与完整独立入口 `PGMTTerrainGlimpse`。默认编码器为 PMT 原生 `MLP(52,128,(128,),"elu")`：隐藏层 ELU、输出层线性，共 **23,296** 个参数。四个 patch 使用同一组权重，直接对 `[N,4,52]` 的最后一维计算；包含 selector 的整个模块共 **440,968** 个参数。隐藏宽度/激活可通过 `PGMTTerrainPatchEncoderCfg` 配置，输出 128 维保持固定；这些配置属于工程选择。

编码输入顺序为归一化高度 25 维、逐点 bool 转为高度 dtype 的有效性 25 维、torso yaw 坐标下的米制 `(x,y)` 2 维。高度/掩膜按行 y、列 x 展平。模块内部不重复归一化、不重新加噪、不修改原观测，也不重新选择无效 patch 的位置。

```python
from motion_tracking_rl.networks.pgmt_terrain_glimpse import PGMTTerrainGlimpse

glimpse = PGMTTerrainGlimpse().to(device)  # 与策略一起构建一次。
# terrain_map 是米制 P2.1 version=3 观测；commands 已经由原有 normalizer 处理。
result = glimpse(terrain_map, terrain_valid, history_token, commands, return_debug=True)
terrain_tokens = result.terrain_tokens  # [N,4,128]，保留 autograd。
token_valid = result.token_valid        # [N,4] bool，必须传给后续 IFM 掩码。
debug = result.debug                   # 默认 return_debug=False 时为 None。
```

`PGMTTerrainGlimpseOutput` 的主输出为 `terrain_tokens`、`token_valid`；`debug` 为可选的 `PGMTTerrainGlimpseDebug`：

| 调试字段 | 形状与语义 |
| --- | --- |
| `center_xy_m` | `[N,4,2]`，实际 torso yaw 坐标下的米制 `(x,y)` |
| `patch_height` | `[N,4,5,5]`，网络归一化 sensor−ground 高度；乘 3 才是米制相对高度，无效点为 −1 |
| `point_valid` | `[N,4,5,5]` bool，逐点有效性 |
| `valid_weight` | `[N,4,5,5]`，有效邻点的插值支持权重 |
| `coverage` | `[N,4]`，有效点数量 / 25 |
| `token_valid` | `[N,4]` bool，与主输出的值一致，但不共享存储 |

调试输出默认关闭以避免训练时额外复制；开启后所有调试字段均在原设备上 `detach().clone()`，不持有训练图、不与主路径共享存储，不自动搬到 CPU，也不缓存在模块成员中。可视化端可以选择环境后自行转 CPU；这些展示副本不能用于训练损失。Isaac 叠加显示仍属 P2.5。

**无效 patch 仍产生有限、可能非零的编码向量，同时 `token_valid=False`。** 不用清零向量代替 attention 屏蔽；全图无效时四个标记均为 false。有效性阈值为离散判定，梯度通过高度和中心连续路径传播。真正从 IFM softmax 排除无效 terrain K/V，以及全无效时仅使用 history/command，由 P2.3 接入。

验证沿用第 3.2 节命令：**153 passed，4 skipped**。新增检查包括 52 维特征顺序、共享参数与 patch 重排、N=1/批量、0/12/13 个有效点的 token 掩膜传递、主路径只预处理一次、重复前向及 train/eval 不消耗 RNG、不修改输入、调试副本无图/无别名且可修改、调试开关前后 token 和参数梯度一致、非默认配置与 state_dict 严格保存重载。将 encoder 第一层的两个中心输入列置零后，token 损失仍经 crop 高度回传 selector 全部层及地图/history/commands，排除了仅由中心拼接产生的旁路梯度。4 项 CUDA 检查因当前进程 CUDA 不可用而跳过。

另用 P2.1 version=3 保存的五类 L0/L5/L9 及 L9 增强地图，共 **20 张实际地图、80 个 patch** 做 CPU 离线前向/反向。在种子 2244、随机初始化、零 history/归一化 command 条件下，80/80 patch 有效，1,965/2,000 个采样点有效；所有 token 有限，selector/encoder 的全部参数均收到有限非零梯度，调试开关不改变结果或 RNG。单独全无效地图也返回有限 token 与全 false 掩膜。这些统计只对应此次输入与未训练中心，不代表策略学会了选点。

**同日 GPU 补验与原因更正：** 前述 CUDA 不可用来自受限测试沙箱未暴露 `/dev/nvidia*`，不是主机显卡或驱动故障。在沙箱外确认两张 RTX 5090、驱动 595.58.03 正常，同一 `pmt` 环境的 PyTorch 2.7.0+cu128 返回 CUDA 可用，实际 GPU 张量运算成功。随后在 `cuda:0` 补跑此前跳过的输入、selector、crop 和完整 token 路径四项前向/反向检查，结果为 **4 passed，106 deselected，0 skipped**。结合此前 153 项已通过回归，这 157 项均已有通过记录；本次没有重新跑 CPU 项。见 [CUDA 补验记录](../logs/pgmt_p22_acceptance/cuda_followup_20261012_164648.json)。复现补测命令（需在能访问 GPU 的终端执行）：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt_terrain_glimpse.py -k cuda -rA
```

记录含配置、输入/源码哈希、分地形覆盖率和各层梯度，见 [encoder 第 4 项离线记录](../logs/pgmt_p22_acceptance/encoder_step4_20261012_163413.json)；中心、裁剪高度、掩膜、权重、覆盖率及 token 已保存到 [调试数据 NPZ](../logs/pgmt_p22_acceptance/encoder_step4_20261012_163413_debug.npz)。第 4 项完成时 CPU/CUDA 功能与梯度检查已通过；后续第 5 项补齐的耗时/显存、中心分布等结果见第 3.6 节。当前没有 IFM/PPO 接入或新的 Isaac 训练，模块级权重重载也不等于 P2.4 训练续训验收。

### 3.6 P2.2 第 5 项：独立验收完成（2026-10-12）

通过 [validate_pgmt_terrain_glimpse.py](../scripts/validate_pgmt_terrain_glimpse.py) 在 CPU 与 RTX 5090 `cuda:0` 完成独立验收，相关回归 **157 passed，0 skipped**。入口显式要求可见 CUDA，完整记录见 [P2.2 验收报告](PGMT_P22_ACCEPTANCE.zh-CN.md)；机器可读结果为 [最终 report.json](../logs/pgmt_p22_acceptance/full_20261012_165716/report.json)。

- 五类地形 L0/L5/L9 及 L9 增强图共 20 张，扩展原样、额外 10% 丢点、中央缺测、地图边界、全无效五种状态；CPU/CUDA 各 30,000 个采样点与独立 NumPy 对照。最大归一化高度误差 `2.56e-7`，点/patch 掩膜全部一致。
- CPU/CUDA token 最大差异 `2.38e-7`；去除直接中心编码贡献后，token 损失仍经裁剪高度回传 selector；含缺测分母导数的有限差分误差约 `4.08e-11`。
- 5 个初始化种子、两种条件，共 40,960 个中心的分布已记录：没有坐标达到 ±0.79 m 边界区，未训练 patch 存在明显重叠，保留为训练期诊断基线。
- batch 1/160/1024/8192 的 CPU/CUDA 前向、调试前向、前向加反向计时和 GPU 显存已记录。8192 下 GPU 前向中位约 0.814 ms、前向加反向约 1.687 ms；后者峰值 PyTorch allocated 约 297.19 MiB。数值仅对应独立模块，测量方式、P95 和范围见验收报告。

**P2.2 第 1–5 项完成。** 接下来 P2.3 负责 terrain K/V 与无效 token 屏蔽、actor/critic 融合和 PPO 梯度；P2.5 再验收实际训练中的中心变化、总显存和吞吐。

### 3.7 P2.3 实施步骤：IFM 接入（第 1–5 部分完成）

**目标：让已经验收的 4 个地形 token 通过同一个 IFM 影响 actor 和 critic，并让原生 PPO loss 能更新 selector 与 patch encoder。** 第 1–5 部分已完成，记录见第 3.8–3.12 节及 [P2.3 融合验收](PGMT_P23_ACCEPTANCE.zh-CN.md)。依据为论文第 3 页图 2、式 (3)–(4)，第 4 页式 (6)–(8)，以及第 5 页 “Integration with IFM”。论文要求扩展 K/V 并共享 intent；以下掩膜、接口和诊断规则为 PMT 工程实现。

#### 现状与改动边界

| 当前代码 | 对 P2.3 的影响 |
| --- | --- |
| [PGMTCrossAttention](../motion_tracking_rl/networks/pgmt_actor_critic.py) 已支持可选 key 掩膜与 attention 调试 | 第 1 部分完成；保留既有投影层参数名、形状及无掩膜路径，后续接入 policy |
| [PGMTActorCritic._encode()](../motion_tracking_rl/networks/pgmt_actor_critic.py) 已支持 disabled 的 9-key / enabled 的 13-key 分支 | 第 2 部分完成；复用原有 history/command 条件和 Terrain-Glimpse，不改变 actor/critic 层宽、Gaussian 和 normalizer |
| [PPO.act()/update()](../motion_tracking_rl/algorithms/ppo.py) 分别调用 actor 和 critic；[RolloutStorage](../motion_tracking_rl/storage/rollout_storage.py) 已保存并重排完整 TensorDict | 第 4 部分已证明米制地图/掩膜正确保存、重排，更新时重算并回传梯度；直接复用算法及 storage，无需修改 |
| 当前 [metadata](../motion_tracking_rl/networks/pgmt_actor_critic.py) 已按是否构建感知模块区分 family，登记 cfg/mode、三组价值头、version=3 契约和观测分组 | 第 3 部分完成网络级登记与严格重建；正式入口与环境契约绑定、跨阶段迁移和完整严格续训属于 P2.4，旧 Stage-1 元数据保持兼容 |

P2.3 保留 `[upper,lower,auxiliary]` 三头及原奖励，不加入新的 reward、课程规则、注意力退火或中心分散损失。P2.4 负责正式任务配置、指定 `model_15000.pt` 的跨阶段迁移、完整续训校验；P2.5 负责 Isaac 可视化与短训。这里的原生 PPO 张量验证不等于 Isaac 训练验收。

#### 固定融合契约

| 项目 | 首版实现约定 |
| --- | --- |
| Query | 原有 history token `[N,1,128]` |
| K/V 顺序 | history 1 个 → command 8 个 → terrain 4 个，共 `[N,13,128]`；同一套既有 K/V 投影，不新增模态专属投影 |
| RoPE | Q=0；K 位置为 `[0, 0,1,3,7,15,31,63,127, 0,0,0,0]`。terrain=0 沿用已登记的工程选择，不把 4 个空间 patch 编成未来时间步 |
| 有效性 | `key_valid = cat(ones[N,9], token_valid[N,4])`，bool，True 表示参与 attention；向 SDPA 广播为 `[N,1,1,13]` |
| 全部 terrain 无效 | 每个环境独立屏蔽末尾 4 项；前 9 项始终有效，不产生全 −∞ 的 softmax 行，也不依赖清零 V 实现屏蔽 |
| intent 与输出 | intent 仍为 `[N,128]`；actor 输出 `[N,29]`，critic 输出 `[N,3]` |
| attention 调试 | 默认 4 heads，输出每个 head 的权重 `[N,4,1,S]`，S=9 或 13；同时提供 key 有效性与顺序，禁止先平均 heads 再隐藏差异 |

**保留原生分开 forward 的调用方式。** actor 与 critic 共享同一个 glimpse/IFM 实例，并消费同一份已经增强、保存的地图和掩膜；critic 的额外 privileged 输入仍在 critic MLP 处拼接，不反向泄露给 actor。固定相同观测和 normalizer 状态，两次 `_encode()` 应得到数值一致的 intent。这里的“共享”指同一函数、参数和输入语义，不要求跨 API 调用复用同一个 tensor。

每次 `_encode()` 内只将原始米制地图预处理一次，重算时仍从原始观测开始，不能把前一次归一化结果再次除以 3。禁止按 tensor 地址、控制步或“最近一次 forward”缓存带图 token，也禁止把 rollout 的 detach token 作为 PPO 的网络输入。暂不为消除两次计算而修改原生 PPO 调用协议；整网成本在后续验收中测量。

#### 五部分实现顺序

1. **✅ 扩展 IFM 的 key 掩膜和 attention 调试。**
   - `PGMTCrossAttention.forward()` 已增加可选 `key_valid[N,S]`，校验 bool、batch、长度和设备。**True=允许参与**，省略掩膜时保留原 SDPA 调用。
   - 新增 `return_debug=True`，返回 `PGMTAttentionOutput(output, debug)`；默认仍返回 Tensor。实际输出始终通过 SDPA 计算，调试从同一份旋转后 Q/K 独立重建逐 head 权重，不保留训练图。
   - CPU/CUDA 已验证无掩膜等价、无效权重/输入梯度为 0、非空行权重和为 1、不同 batch 掩膜、调试无副作用，以及四个 terrain 全无效时与仅用原 9 个 key 等价。正式 key 拼接和掩膜构造已由第 2 部分接通。

2. **✅ 在 `_encode()` 中接通 Terrain-Glimpse 与 13 个 K/V。**
   - 将已有 history attention 输出和 command normalizer 输出直接传给 `PGMTTerrainGlimpse`；读取同一份 `terrain_map` / `terrain_valid`，不另建 encoder、normalizer 或增强逻辑。
   - 将 `terrain_tokens`、`token_valid` 分别用于 K/V 拼接和 key 掩膜，补齐 RoPE 位置。启用感知时要求 embedding_dim=128；旧无感知实例保持现有可用配置。
   - 显式区分模块配置和运行模式：`terrain_glimpse_cfg=None` 表示原 Stage-1、没有新增参数；配置字典（空字典可表示 P2.2 默认值）才构建 `terrain_glimpse`，`terrain_mode` 为 `disabled/enabled`。默认保持原 Stage-1 行为；感知配置与模式组合错误直接拒绝。
   - `disabled` 跳过 glimpse，只用原 9 个 K/V；已构建的感知模块可以保留在对象中用于等价回归切换。`enabled` 使用 13 个 K/V 和逐环境掩膜。模式不在一次 rollout/PPO 更新途中隐式改变；初版有效 terrain 的 logit bias=0。
   - 已验证上述构造和切换、N=1/混合掩膜批次、实际拼接与 NumPy attention 对照、一次预处理和通过 crop 的 selector 梯度。第 3 部分补全显式观测分组和整网诊断，第 4–5 部分已完成原生 PPO 及真实地图融合验收。详见第 3.9–3.12 节。

3. **✅ 对齐 actor/critic、观测接口与元数据。**
   - `_update_distribution()`、`act_inference()`、`evaluate()` 统一走同一融合实现。actor/critic 使用同一份增强地图；不为 critic 新开干净地形通道，亦不重复采样。
   - 在感知实例的 `obs_groups` 中显式登记独立的地图/掩膜项（建议 `terrain_map:[terrain_map]`、`terrain_valid:[terrain_valid]`），保持二维形状，不拼入 96 维 proprio 或 critic 原始向量。缺少启用模式所需的字段、形状或 dtype 不匹配时明确报错。
   - 保留全部旧模块的 state_dict 前缀，新参数集中在 `terrain_glimpse.*`。无感知实例保持旧签名；感知实例登记 mode、三组 value 名称、P2.1 version=3 输入契约及 selector/crop/encoder 配置，供 P2.4 迁移和严格续训使用。元数据登记本身不算完成 checkpoint 迁移。
   - 提供可选 `get_perception_debug(obs)` 或等价诊断入口，返回中心、patch、有效性、每 head attention、terrain attention 总占比。全部为 `detach().clone()` 副本，不更新分布、normalizer 或 RNG，不保存训练图；关闭时不生成大调试副本。
   - CPU/CUDA 已验证相同增强输入与重算、特权信息不进入 actor 前向、两条损失经 crop 到共享 selector 的梯度，以及调试副本隔离；263 项相关测试通过、无跳过。详见第 3.10 节。

4. **✅ 验证原生 PPO 的观测保存、重算与梯度。**
   - 原生 PPO/storage 已完成连续两轮短 rollout、三头 returns、多 epoch/minibatch 和真实 Adam step；覆盖有效、边界缺测/增强、全无效地形及全局/每 minibatch 优势归一化。
   - 样本标识证明 map/mask 与动作、旧 log probability、value/returns 等共同保存和重排；更新期间禁用观测重取与增强入口，输入仍精确匹配保存值。
   - 原生策略目标、value 目标、完整 PPO loss 分别通过；首次反传前置零 encoder 中心坐标输入列后，两条损失仍经裁剪高度回传 selector，selector/encoder/IFM 梯度有限且真实更新。
   - `disabled` 或四个 terrain 全无效时，对动作/价值无贡献，terrain 梯度为零或 None。CPU/CUDA 共新增 22 项验收、完成 240 次 Adam step；总计 285 项相关回归通过、无跳过。无需改动 PPO/storage，详见第 3.11 节。

5. **✅ 完成 CPU/CUDA 融合验收并记录。**
   - 新增 `scripts/validate_pgmt_ifm.py`，复用 P2.1/P2.2 的 20 张真实地图与已有 IFM/PPO 回归；每个设备验证 100 个地图变体、全部 N=1 与 N=100 混合掩膜批量。覆盖五类 L0/L5/L9、L9 增强、额外丢点、中央缺测、半图边界和全无效。
   - 显式复制并逐项检查 49 个旧 state_dict 项（含全部 normalizer），只新增 10 个感知项，完整字典 `strict=True` 加载。disabled 的动作/value/intent 与旧路径逐值一致，全无效回退误差 ≤5.96e-8，均通过 `atol=rtol=1e-5`。
   - 仅改变有效地图时三个输出全部响应；在 IFM 输入处改动被屏蔽 token，输出逐值不变。没有把这个结论错误扩展到 selector 读取的原始无效高度。
   - 正常/调试输出一致，每 head 的无效权重为零，掩膜 CPU/CUDA 精确相同；输入、RNG 与 normalizer buffer 不变，无 NaN/Inf；旧 Stage-1 等 285 项回归通过、无跳过。
   - 完整 actor/critic `[512,256,128]` 的 batch 160/1024 前向及原生 PPO 成本已记录，性能基准额外完成 384 次 Adam step。GPU 原有并行高负载已明确注明，不将其作为独占吞吐结论。详见第 3.12 节与独立验收记录，P2.3 完成。

#### 完成标准与代码落点

| 完成标准 | 证据 / 落点 |
| --- | --- |
| 13 个 K/V、RoPE 和有效性掩膜正确 | `pgmt_actor_critic.py`；每 head attention 与独立手算对照 |
| 关闭感知或全无效地形可回到旧行为 | 固定权重、normalizer、观测的 CPU/CUDA 等价回归 |
| actor/critic 使用同一观测及感知参数 | `_encode()` hook、输入一致性和两条 loss 的梯度检查 |
| PPO 更新实际训练感知模块 | 原生 `PPO` / `RolloutStorage` 张量集成测试、参数更新前后对照 |
| 原 Stage-1 兼容、诊断无副作用 | 既有测试 + `tests/test_pgmt_ifm.py`；新配置元数据可重建感知实例 |
| 结果可复现 | 融合验收脚本、配置/源码哈希、数值/梯度/性能报告；正式配置和训练启动放在 P2.4 |

### 3.8 P2.3 第 1 部分实施记录（2026-10-12）

修改 [PGMTCrossAttention](../motion_tracking_rl/networks/pgmt_actor_critic.py)，新增 `PGMTAttentionDebug` 与 `PGMTAttentionOutput`。没有新增可学习参数、persistent buffer 或训练图缓存，`q/k/v/out` 的 state_dict 名称与形状不变。当前 `PGMTActorCritic._encode()` 仍调用原 9 个 key，实际 terrain token 注入留给第 2 部分。

```python
# query [N,L,D]；keys [N,S,D]；q/k_positions 为原有 RoPE 位置。
output = attention(query, keys, q_positions=q_positions, k_positions=k_positions,
                   key_valid=key_valid)  # 默认返回 Tensor [N,L,D]。
result = attention(query, keys, q_positions=q_positions, k_positions=k_positions,
                   key_valid=key_valid, return_debug=True)
output = result.output  # 主计算图保留梯度。
weights = result.debug.weights  # [N,heads,L,S]，逐 head，展示副本。
```

| 接口 | 实现语义 |
| --- | --- |
| `key_valid` | 可选 bool `[N,S]`，必须与 query/keys 同 batch/device，S 与 key 数一致；不接受隐式 batch 广播、浮点掩膜或外部 4D mask。True 表示允许参与，内部广播为 `[N,1,1,S]` |
| 默认路径 | `key_valid=None` 时沿用原 SDPA 调用，默认返回 Tensor；原有 history/intent attention 调用不变 |
| `debug.weights` | `[N,heads,L,S]`，使用真实投影并 RoPE 旋转后的 Q/K、`1/sqrt(head_dim)`、相同掩膜计算 softmax；列顺序即传入 keys 顺序，不平均 heads |
| `debug.key_valid` | 独立 bool `[N,S]` 副本；省略输入掩膜时为全 true |
| `debug.q_positions/k_positions` | 实际应用的 RoPE 位置副本；未启用 RoPE 时均为 None |
| 调试副本 | `no_grad` 下计算，`detach().clone()`，保留在计算设备，不共享输入存储、不缓存计算图；不会改变主输出、梯度、RNG 或 train/eval 行为 |

主路径始终调用 `scaled_dot_product_attention`，调试权重重建不替代主路径。FP16/BF16 的调试分数提升到 FP32，并关闭该小段的 autocast，避免大分数在 half 矩阵乘法中溢出；float64 保持 float64。融合内核与显式分数计算可能存在正常浮点舍入差异，因此调试权重是数值对应的解释，不承诺逐位复原融合内核的内部中间值。

**四个 terrain key 全无效**时，只要前 9 项为 true，其输出与移除末尾 4 项后的 attention 等价。低层通用接口也定义了**所有 key 全无效**的极端情况：注意力聚合为零，输出为既有 `out.bias`，调试权重全零、query/key 输入梯度为零，不生成 NaN；调试实现通过张量操作处理空行，不增加 CPU 同步判断。此极端情况不代表 IFM 的正常回退，正式 IFM 必须始终保留 history/command。

新增 [test_pgmt_ifm.py](../tests/test_pgmt_ifm.py)，覆盖掩膜校验、无掩膜旧路径、独立 NumPy 投影/RoPE/softmax 对照、批内不同掩膜、多个 query、修改已屏蔽 token 后输出不变、无效 key 梯度为零、9-key 回退、调试副本修改不影响 backward、输入/RNG/state_dict 不变，以及 CUDA FP16/BF16 大分数的有限性。

在主机 RTX 5090 / PyTorch 2.7.0+cu128 运行相关回归：**188 passed，0 skipped**（新增 attention 检查 31 项，其中 CUDA 13 项）。FP32 独立对照的输出最大绝对误差约 `1.42e-7`、权重约 `7.94e-8`，屏蔽末尾 4 项与只用 9 个 key 的最大误差约 `8.94e-8`；float64 对照误差不超过 `3.89e-16`。

复现命令（需要能访问 GPU）：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_ifm.py tests/test_pgmt_terrain_glimpse.py tests/test_pgmt_perception.py \
  tests/test_pgmt_rl.py tests/test_normalization_resume.py
```

记录含源码哈希与 CPU/CUDA 数值结果，见 [第 1 部分 report.json](../logs/pgmt_p23_acceptance/ifm_step1_20261012_214825/report.json) 和 [测试日志](../logs/pgmt_p23_acceptance/ifm_step1_20261012_214825/pytest.log)。本次只完成 attention 层能力；尚未完成 policy 中的 terrain K/V 构造、actor/critic 感知接入或 PPO 感知梯度验收，也没有启动 Isaac。

### 3.9 P2.3 第 2 部分实施记录（2026-10-12）

已在 [PGMTActorCritic](../motion_tracking_rl/networks/pgmt_actor_critic.py) 中接通可选 Terrain-Glimpse。每次 `_encode()` 取既有 history attention 输出和 command normalizer 输出，连同当前 TensorDict 中米制 `terrain_map` / 配对 `terrain_valid` 传给同一个 `PGMTTerrainGlimpse`。地图只在其输入层 `clip(-3,3)/3` 一次；不重采样增强、不缓存 token、不 detach 训练分支。新模块在所有既有层之后构建，保留相同种子下原有参数的初始化值与 state_dict 前缀。

| 构造参数 | 行为 |
| --- | --- |
| `terrain_glimpse_cfg=None, terrain_mode="disabled"`（默认） | 原 Stage-1：无新增感知参数，不读取地图，只使用原 9 个 key；原无感知 embedding 配置继续可用 |
| `terrain_glimpse_cfg={}, terrain_mode="enabled"` | 构建 P2.2 默认模块（440,968 个参数），embedding 必须为 128；使用 13 个 key |
| `terrain_glimpse_cfg={...}, terrain_mode="disabled"` | 保留感知模块用于回归，但不执行 glimpse，也不要求提供地图；可显式改为 enabled |
| 配置或模式错误 | 拒绝未知模式、未构建模块却启用感知、非字典/未知配置、非 128 的感知 embedding；缺失地图/掩膜报错，形状/设备/dtype 由既有 adapter 校验 |

配置字典可包含 `selector_config`、`crop_config`、`encoder_config` 三个子字典；省略项沿用 P2.2 默认值。例如：

```python
terrain_glimpse_cfg = {
    "selector_config": {"hidden_dims": [256, 128], "center_limit_m": 0.8},
    "crop_config": {"min_valid_weight": 0.5, "min_valid_points": 13},
    "encoder_config": {"hidden_dims": [128], "activation": "elu"},
}
terrain_mode = "enabled"
```

K/V 严格按 `[history×1, command×8, terrain×4]` 拼接；RoPE 位置为 `[0,0,1,3,7,15,31,63,127,0,0,0,0]`，query 位置为 0。`key_valid=cat(ones[N,9], token_valid[N,4])`，前 9 项始终 true；有效地形没有额外 logit bias。全无效地形逐环境退回原 9-key attention，intent 保持 `[N,128]`。actor/critic 既有入口会自然调用这一 `_encode()`，但完整接口与诊断验收仍属于第 3 部分。

为避免感知实例误标 Stage-1，本部分提前登记最小元数据：构建感知模块的实例标记 `pgmt_perception_only`（包括暂时 disabled 的实例），`policy_kwargs/signature` 记录完整模块配置和当前 mode；原 Stage-1 family/kwargs/signature 保持原格式。配置字典不与调用者共享可变列表。已验证按这些构造信息重建网络并 `strict=True` 重载 state_dict；这只是网络级重建，不是指定 Stage-1 checkpoint 迁移或完整训练续训验收。

**验收：CPU + RTX 5090，216 passed、0 skipped**，其中本次新增 28 项（CUDA 7 项）。见 [最终 report.json](../logs/pgmt_p23_acceptance/ifm_step2_20261012_223836/report.json)、[pytest 日志](../logs/pgmt_p23_acceptance/ifm_step2_20261012_223836/pytest.log) 和 [JUnit 记录](../logs/pgmt_p23_acceptance/ifm_step2_20261012_223836/pytest.xml)。

- N=1 和混合掩膜批次覆盖全有效、部分有效、全无效环境；hook 验证实际 token 来源/顺序、RoPE 和 key mask。独立 NumPy attention 对照最大 intent 误差 `5.14e-8`。
- 相同原有权重/normalizer/输入下，disabled 与原 9-key 路径逐位一致；enabled 全无效地形的 intent 最大误差 `2.98e-8`，动作均值与三头 value 对照通过 `atol=1e-6, rtol=1e-5`。此处采用固定种子的等权重网络，**未加载实际 `model_15000.pt`**。
- 每次 `_encode()` 只调用一次 glimpse 和地图预处理；条件来自既有 history/command 输出；输入、normalizer 统计、state_dict 和 RNG 不变。
- 有效非平坦地图能改变 intent；将 patch encoder 的中心坐标输入权重置零后，intent 损失仍经裁剪高度回传 selector，梯度有限且非零。disabled/全无效时，感知参数与地图梯度为零或 None；本次没有使用 PPO loss 或更新感知参数。
- 验证中修正了测试夹具的 TensorDict 键遍历方式；单环境用独立批次校准 normalizer，避免以单个样本的零方差制造不必要的数值放大。保留原数值容差；失败的中间记录保留在日志目录。

复现命令与第 3.8 节相同。**本记录对应第 2 部分完成时的状态**；后续观测分组、元数据、actor/critic、整网调试、原生 PPO 与真实地图融合验收见第 3.10–3.12 节。调用者应仅在 rollout/update 周期之间显式切换模式，网络不会自动切换，也不持有 PPO 周期状态；本部分未修改正式训练配置、未启动 Isaac。

### 3.10 P2.3 第 3 部分实施记录（2026-10-13）

**已完成 actor/critic 共享输入、网络观测分组、感知元数据与整网调试接口。** 核心实现见 [pgmt_actor_critic.py](../motion_tracking_rl/networks/pgmt_actor_critic.py)。critic 仍使用 251 维特权观测经原归一化器处理后拼接 128 维 intent；未来多帧 command 仍通过 IFM 提供，输出保持 upper/lower/auxiliary 三头。

**观测接口。** 凡构建 `terrain_glimpse` 的实例（包括暂时 disabled 的实例），必须显式提供两个独立分组：

```python
obs_groups = {
    "policy": ["proprio"],
    "policy_history": ["proprio_history"],
    "command_window": ["command_window"],
    "critic": ["critic"],
    "terrain_map": ["terrain_map"],
    "terrain_valid": ["terrain_valid"],
}
```

每个地形分组必须且只能指定一个字段，可映射到其他保存字段名；读取严格遵循映射，不回退到硬编码名称。两者不能互用同一字段，也不能混入 policy/critic/history/command 分组。地图与掩膜保留 `[N,21,21]`，启用模式在构造和前向时校验 batch、shape、device、dtype：地图必须与 proprio 共享浮点 dtype，掩膜允许 bool 或浮点（adapter 统一按 `>0.5` 判有效）。缺少字段会报告分组和实际字段名。disabled 不读取地图，允许纯网络回归时省略地图张量；原生 runner 的观测分组解析仍会检查配置中声明的字段。

`_update_distribution()`、`act_inference()` 和 `evaluate()` 使用同一个 `_encode()`，共享同一个 glimpse/IFM 实例。验证使用 P2.1 实际增强函数生成的原子地图/掩膜，三个入口读取相同 tensor，intent 数值一致且各自拥有新的计算图；原址修改保存地图后会重新计算，不返回旧 token。actor/critic 分开 forward，各调用一次地图预处理，不共享跨 forward 的训练图。critic 专用特权输入只拼入 critic MLP；critic 的损失可以训练共享感知参数，但特权向量不会进入 actor 前向。

**整网调试。** 新增 `PGMTPerceptionDebug`，用 `policy.get_perception_debug(obs)` 获取；调用会额外执行一次融合前向，但不调用 actor/critic MLP、不创建新动作分布、不采样动作、不切换 train/eval、不更新 normalizer/RNG。

| 调试字段 | 内容 |
| --- | --- |
| `intent` | `[N,128]` 的融合结果独立副本（旧无感知实例可使用其原 embedding 宽度） |
| `glimpse` | enabled 时包含 `center_xy_m[N,4,2]`、`patch_height[N,4,5,5]`、点有效性、插值权重、覆盖率与 token 有效性；disabled/Stage-1 为 None |
| `attention.weights` | `[N,heads,1,S]`，保留每个 head，S=13 或 9；无效 key 权重为零 |
| `attention.key_valid/q_positions/k_positions` | 实际 key 掩膜和 RoPE 位置副本 |
| `terrain_attention_mass` | `[N,heads,1]`，对最后 4 个 terrain key 的权重求和；disabled 或所有 terrain 无效时为零，不平均 heads |
| `key_labels` / `terrain_mode` | 每个 attention 列的 history/command[offset]/terrain[index] 标签及本次模式 |

中心单位 m；patch 高度已归一化，乘 3 才是米制相对高度，点无效处仍填 −1。所有调试 tensor 均为设备内的 `detach().clone()` 独立副本，无训练图、无持久缓存；修改后再查询不会污染原数据或待反传计算图。普通前向不生成这些副本。

**元数据。** 新增可选构造参数 `terrain_observation_cfg`，用于声明输入对应的完整 P2.1 version=3 配置；省略时使用当前默认契约。将原有纯配置函数和扫描常量集中到 [pgmt_perception_contract.py](../motion_tracking_rl/pgmt_perception_contract.py)，环境感知与网络共同导入；原 `pmt_tasks.pgmt_perception` 导入接口保留，实际高度处理/增强顺序没有改变。

- `policy_kwargs` 保存当前 mode、selector/crop/encoder 配置及声明的 observation contract，可据此重建网络。
- `perception_contract` 和 `signature` 保存 version=3 的符号、米制单位、世界 Z=30 m 填充、网络归一化位置、丢点/噪声参数及采样时机；同时记录网络地图分组、mask 语义、K/V 顺序、RoPE、零 logit bias 和 actor/critic 共享输入约定。三组 value 名称也进入感知实例 signature。
- 构造时拒绝 version=1/2、反转高度符号或改为观测侧归一化等不兼容声明。配置、观测分组和返回的元数据均作独立副本，调用者修改不会静默改变网络契约。
- 原 Stage-1 的 family、kwargs/signature 格式及 state_dict 前缀保持兼容；新增权重仍全部在 `terrain_glimpse.*`。已验证自定义契约/分组下的网络重建与 `strict=True` 权重重载、动作/价值一致。

这里记录的是**声明的输入契约**，无法只从 TensorDict 的数字判断符号或增强来源。P2.4 必须将该参数绑定到实际 `env.cfg.pgmt_terrain_observation`，并补齐严格续训检查；本部分没有把默认契约自动视为已核实的环境配置，也没有改变 runner 当前的加载策略。

**验收：263 passed、0 skipped，CPU/CUDA 全部通过**；新增 47 项，IFM 文件合计 106 项（其中 CUDA 33 项）。记录见 [report.json](../logs/pgmt_p23_acceptance/ifm_step3_20261013_123512/report.json) 和 [pytest 日志](../logs/pgmt_p23_acceptance/ifm_step3_20261013_123512/pytest.log)。验证涵盖增强输入一致性与原址重算、两条损失对共享模块的梯度、特权输入隔离、观测重命名/错误接口、契约版本拒绝和重建，以及 train/eval × enabled/disabled/全无效/Stage-1 下的调试副本、梯度、分布、RNG 和归一化统计不受影响。

本部分的梯度检查使用 actor 输出和 critic value 的张量损失，屏蔽 encoder 中心坐标直连后仍能经裁剪高度回传 selector；原生 PPO 感知更新与真实地图/性能后续验收见第 3.11–3.12 节。本部分未启动 Isaac，也未执行实际 `model_15000.pt` 迁移。

### 3.11 P2.3 第 4 部分实施记录（2026-10-13）

**原生 PPO 的观测保存、重算与梯度验收通过，无需修改 PPO、RolloutStorage 或策略实现。** 新增 [test_pgmt_perception_ppo.py](../tests/test_pgmt_perception_ppo.py) 与可复现入口 [validate_pgmt_perception_ppo.py](../scripts/validate_pgmt_perception_ppo.py)。测试使用合成非平坦射线命中，调用实际 P2.1 高度处理、原子打包及拆分接口；未启动 Isaac。

CPU 与 RTX 5090（PyTorch `2.7.0+cu128`）完整运行结果为 **285 passed，0 skipped，20.87 s**。其中本部分新增 **22 项（CPU 11、CUDA 11）**，共执行 **240 次真实 Adam 参数更新**。证据包括 [report.json](../logs/pgmt_p23_acceptance/ifm_step4_20261013_132541/report.json)、[pytest.log](../logs/pgmt_p23_acceptance/ifm_step4_20261013_132541/pytest.log) 与 [JUnit](../logs/pgmt_p23_acceptance/ifm_step4_20261013_132541/pytest.xml)，报告保存源码哈希、配置和逐用例梯度/参数变化量。

**验证配置。** 6 个环境、rollout 长度 4，每轮更新 2 epochs × 3 minibatches，每个 minibatch 8 个样本，固定 Adam 学习率 `3e-4`。启用感知的用例连续执行两轮 rollout/update，以检验 storage 复用与计算图重建；回退用例执行一轮。保留默认 128 维 embedding 和完整 Terrain-Glimpse，actor/critic 隐藏层缩为 `[32,16]`，三价值头及 29 维动作不变。该规模用于接口和梯度验证，不用于吞吐结论。

| 检查 | 结果与证据 |
| --- | --- |
| 原始观测保存 | storage 保存 `[T,N,21,21]` 米制地图和配套浮点掩膜；保存后原址覆盖旧输入，storage 仍逐值等于原快照。仅保存原始观测，没有把 detached terrain token 写入 rollout |
| flatten / shuffle | 测试专用 `sample_id` 跟踪每个样本；每个 minibatch 的全部观测、动作、旧 value/log probability、mu/sigma、returns/advantages 均匹配。每个 epoch 恰好覆盖所有样本，无地图/掩膜错配 |
| 三组 GAE | 分别覆盖真实终止与 timeout，核对 timeout 的三头奖励 bootstrap，并用逐环境/逐价值头的独立标量递推对照 returns；actor advantage 仍按三组求和。全局归一化和每 minibatch 归一化均通过 |
| 更新时重算 | rollout 在 `inference_mode` 下执行；更新恢复 autograd。断开观测源与高度处理/打包/拆分入口后仍完成 PPO 更新。每个 minibatch 的 actor、critic 各重建一次 glimpse/crop/IFM 计算图，输入精确等于保存数据，intent 相同 |
| 一次预处理 | 每次 actor/critic forward 各调用一次 adapter，实际输出精确等于保存米制地图 `clip(-3,3)/3`；不重复加噪、不在保存值上累计除以 3。更新期间所有 normalizer buffer 保持不变 |
| 独立策略/价值目标 | actor-only 使用原生 clipped surrogate，`value_loss_coef=0, entropy_coef=0`；critic-only 使用原生 `value_only_warmup_iters`，不冻结共享模块；combined 使用原生 surrogate + 三头 clipped value loss + entropy。每种目标均执行真实 Adam 更新；独立目标对应的另一个输出 MLP 不变 |
| 经过裁剪的梯度 | 首次反传前将 patch encoder 第一层最后两列（中心坐标输入）置零；actor/critic 的中心、patch height、token 梯度均有限且非零，selector/encoder/IFM 参数发生实际变化。首步证明梯度经过 crop 高度，后续 Adam 正常训练所有参数 |
| 无效 token 与回退 | 混合 batch 中被屏蔽 token 的梯度精确为零；全无效输入与移除地形后的动作/价值在 `atol=1e-6, rtol=1e-5` 内一致，改动无效地图不影响输出。disabled 跳过 glimpse；两种回退的 terrain 梯度为零或 None，既有 IFM/actor/critic 仍更新 |

有效用例首次 Adam step 前、原生梯度裁剪后的模块梯度 L1 范围如下；各用例具体数据保存在 JSON 中。

| 模块 | 最小 L1 | 最大 L1 |
| --- | --- | --- |
| selector | 0.06105 | 0.28965 |
| patch encoder | 0.30338 | 1.93125 |
| IFM | 3.79749 | 8.23761 |

**接入约束。** 原生 `PPO.act()` 先保留当前观测引用，在 `process_env_step()` 中由 storage 复制；两者之间调用方不能原址覆盖旧观测。当前 PGMT 打包使用新建的 `torch.cat` 输出，wrapper 仅从该次输出拆分配对视图，本次验收也使用这条实际打包/拆分路径。storage 的随机索引在一次 generator 内生成一次，各 epoch 复用该排列；测试按此原生行为核对完整覆盖，没有修改采样策略。rollout 更新归一化统计，PPO 重算使用更新后的当前统计，因此不把跨 rollout 的 log probability 逐位相等设为验收要求。

回退用例使用全新 Adam，因此同时验证了 terrain 参数不变；若优化器已有过去的动量，仅凭当前梯度为零不能保证参数完全不移动。这里的通过结论是无效地形不贡献当前动作/价值和梯度，不是关闭感知后自动冻结历史优化器状态。

复现命令（需要可见 CUDA；脚本会拒绝静默跳过 GPU 用例）：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_perception_ppo.py
```

**本节为第 4 部分记录；第 5 部分的后续完成情况见第 3.12 节。** P2.4 的指定 `model_15000.pt` 迁移和严格续训、P2.5 的 Isaac 实际训练仍待实施；本次测试不替代这些验收，也不代表已经学会地形适应。

### 3.12 P2.3 第 5 部分实施记录（2026-10-13）

**CPU/CUDA 整网融合验收通过，P2.3 第 1–5 部分全部完成。** 独立入口为 [validate_pgmt_ifm.py](../scripts/validate_pgmt_ifm.py)，完整结果、复现命令、误差和性能表见 [P2.3 融合验收记录](PGMT_P23_ACCEPTANCE.zh-CN.md)，机器报告为 [report.json](../logs/pgmt_p23_acceptance/fusion_20261013_144840/report.json)。网络与原生 PPO/storage 无需修改。

- 复用 20 张真实保存地图，扩展为 100 个样例；CPU/CUDA 分别检查每个 N=1 和混合 N=100 批量，覆盖全有效、部分缺测与全无效。
- 严格复制 49 项旧权重/归一化状态，保留 10 项新感知参数。disabled 与旧网络逐值一致；全部 terrain 无效的回退最大误差 `5.96e-8`。完整 enabled 网络的 CPU/CUDA intent、动作均值、value 误差分别为 `8.94e-8 / 1.42e-7 / 1.49e-7`，满足 `atol=rtol=1e-5`。
- 改变有效地图能影响三个输出；修改已屏蔽的 terrain token 不影响输出。正常/调试一致，逐 head 无效权重为零，输入/RNG/normalizer 统计不变。NumPy 核对 20,000 个裁剪点，最大归一化高度误差 `2.43e-7`，掩膜完全一致。
- 完整 actor/critic 隐藏层 `[512,256,128]`，已记录 batch 160/1024 的两条前向、顺序合计及原生 PPO 更新成本。启用感知的 CUDA PPO allocated 峰值约 `76.99 / 318.92 MiB`（4 步 rollout、2×4 minibatch 更新，包含已预热 Adam；不包含 Isaac）。GPU 开始前已有 90% 以上并行负载，耗时仅表示当时共享负载下的测量，不能据此推算独占吞吐。
- 相关回归 **285 passed、0 skipped**，内含 240 次 Adam 更新；整网性能基准再完成 384 次，总计 624 次。没有启动 Isaac 或迁移 `model_15000.pt`，未改奖励与课程。后续进入 P2.4。

## 4. 感知启用与权重迁移

### 4.0 P2.4 / P2.5 执行安排（2026-10-13，已执行）

1. **正式配置与初始化。** 新增 `pgmt_stage2` 任务、独立 obs/network 配置和 PGMT 专用 runner；绑定实际环境 version=3 感知配置。默认从指定 `model_15000.pt` 逐项迁移旧参数/normalizer，新增感知参数保持初始化，新 Adam、iteration=0、新动作 sampler/课程；保存源哈希与迁移清单。保留三组奖励，不启用 P3。
2. **严格续训。** 单独的 `--resume` 模式，校验完整策略 schema、运行语义、环境身份和 optimizer 后再加载。保存 completed/next iteration、PPO 当前学习率、模型、optimizer、sampler/课程和 Python/NumPy/Torch RNG；恢复 RNG 后重开物理回合，不增加虚假 outcome。新进程验证两次保存/加载及不兼容输入拒绝。
3. **纯张量验收。** 使用实际 Stage-1 checkpoint 检查复制清单、关闭感知等价、优化器初始状态；校验严格恢复、损坏状态拒绝与 RNG。继续运行既有 IFM/PPO 回归。
4. **Isaac 160 环境。** 五类各 32 个，初始 L0–L1，感知启用并联合训练 25 次 PPO 更新；保留已验收 Sim 6/Fabric/碰撞栈配置。审计 reset、参考变换、接触缓存、sampler 单次结算、地图范围和感知梯度；保存逐轮 loss、KL、有效 patch/attention、中心变化和吞吐。
5. **可视化与边界。** 提供全局/跟随/tile 相机，叠加当前保存观测的有效扫描点、4 个中心及 patch 边框。分别检查低/中/高等级、跨 tile 与地图边缘，调试不额外采样观测；保存 GUI 图片和诊断数据。
6. **恢复与规模。** 160 环境新进程续训后再保存/加载，逐项核对恢复状态；进行盲走、仅扫描、完整感知的匹配规模成本对照，并在显存允许时检查 1024 环境。有限验收完成后关闭所启动的 Isaac，保留复现命令和结果；不自动开启长期正式训练。

本轮不新增 attention 退火或地形奖励。若遇到实际问题，先记录触发条件和最小修复，再继续对应验收。耗时同时记录 GPU 并行负载，避免将共享设备测量当作独占性能。

### 感知启用方式

提供 `disabled / enabled` 两种明确模式。`disabled` 必须移除 terrain K/V 或在 attention logits 中严格屏蔽，使 softmax 只包含原有 9 个 token；零地图或仅将 terrain V 置零均不能保证等价。该模式用于回归，不用于训练 selector。

`enabled` 默认联合训练 selector、patch encoder 和原有网络，terrain attention bias 为 0。先完成等价性检查，再在平地及 L0–L1 做小规模短训。若新 token 导致明显 KL/动作跳变，再启用**可选**的负 logit bias 退火，最终归零；论文没有公开其日程，不能把它作为论文要求。退火值在一次 rollout 及其 PPO 更新期间固定，只在下一次 rollout 前改变，并保存进 checkpoint。

### 初始化与续训分开

| 模式 | 处理 |
| --- | --- |
| 默认：Stage 1 → P2 初始化 | 来源仍为用户指定的 `logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt`。复制全部既有网络、三头 critic、探索标准差及 normalizer；只初始化新增感知模块。新优化器、iteration=0，v6 课程和 862 个动作的 sampler 重新初始化 |
| 可选：P1 v6 → P2 warm start | `logs/pgmt_v6_blind/20261011_02/model_15999.pt` 只作显式选择的对照/热启动，不自动替代指定源。新增感知仍属结构迁移，优化器和 iteration 重建；当前 `--initialize-from` 只迁移网络，不提供带入 P1 sampler/课程的选项；持久环境状态只通过 P2 `--resume` 严格恢复 |
| P2 → P2 同配置续训 | 恢复完整模型/normalizer、优化器及其 LR、iteration、sampler 四组统计和更新时钟、课程等级/统计、感知配置与可选注入日程；恢复后重开物理回合，不额外记一次 outcome。保存 RNG 状态并明确恢复时点，不承诺 PhysX 中断轨迹逐位相同 |
| P3 四头扩展 | 按名称将旧 `[upper,lower,auxiliary]` 映射至新 `[upper,lower,terrain,auxiliary]` 的 `[0,1,3]`，terrain 行零初始化；与 terrain 奖励一起验收。完整配置的正式训练仍从指定 Stage 1 源迁移，P2 短训产物作为调试证据 |

新增显式迁移函数与迁移报告，列出复制/新增 tensor、源哈希、观测 schema 和 value group 顺序。不能用宽泛的 `strict=False` 掩盖缺失参数，也不能把普通 `runner.load()` 的元数据警告当作完成结构迁移。对 P2 同配置续训，地图符号、轴序、中心/掩码规则、网络结构及训练配置不匹配应拒绝，而非静默继续。

## 5. 验收安排

| 层次 | 必查内容 |
| --- | --- |
| 张量与几何 | 常数平面、x/y 单轴斜坡、台阶、四个 yaw、地图边缘及全无效图；验证中心坐标、5×5 高度、有效 mask、有限输出。覆盖 N=1 和批量；在有效非平坦样本上检查 selector/patch encoder/IFM 的梯度，并用有限差分单独核对裁剪高度对中心的导数，避免只验证到中心编码的旁路梯度 |
| 迁移回归 | 固定同一 obs、normalizer 状态与推理模式，屏蔽 terrain 后对比确定性动作和旧三头价值，float32 目标 `atol/rtol ≤ 1e-5`；新增地图/掩码 schema 单独核对。不能用各跑一段随机仿真替代同输入比较 |
| 感知路径 | 固定 proprio/history/commands，仅改变有效地图，检查裁剪和网络输出的响应、PPO loss 到感知参数的梯度；这只证明路径接通，不作为已学会选落脚点的证据 |
| 真实短训 | 建议 160 环境（每类 32）、L0–L1、25–50 次 PPO 更新；低/中/高等级另做受控前向与可视化，十级无需在短训内自然升级完。核对奖励组/GAE、有限损失、旧参考不漂移、reset 后扫描刷新及 sampler 每回合只更新一次 |
| 二次恢复 | 保存 → 新进程加载 → 继续更新 → 再保存/加载；逐项比较模型/normalizer、优化器/LR、sampler、课程、iteration 和日程。另做噪声打开的短回归，检查环境观测函数重复调用会重新采样，而同一份已保存观测的网络/PPO 重算保持确定性 |
| GUI 检查 | 复用已有全局/跟随/tile 相机切换，叠加扫描有效区域、4 个中心和 patch 边框，显示 patch 高度/掩码、各 head 的 terrain attention；统计中心重叠、边界饱和与有效 patch 比例。世界坐标由实际 torso yaw 变换，调试绘制可关闭且不修改策略输入 |
| 吞吐与显存 | 相同环境数、rollout 和硬件下，比较盲走、仅开启扫描、开启完整感知的采样耗时与 PPO 更新耗时，记录新增显存；先通过 160，再按显存允许测试 1,024。以当前 float32 storage 估算，8,192×24 的地图和掩码本身约增加 0.65 GiB，尚未计激活；不直接上满规模 |

正式启用时不默认冻结旧主干；学习率先配置化并记录，是否调整依据短训 KL 与平地回归。此次验收证明感知可训练、可续训、可检查；地形适应性能与完整 PGMT 复现结论需要 P3/P4。

## 6. 建议代码落点

`pmt_tasks/env_cfgs/pgmt/stage2.py`、感知网络、严格 checkpoint 和专用训练入口均已实现；实际验收范围及扩容限制见 P2.4/P2.5 报告。P2.1 的版本契约及无缓存增强实现位于 `pmt_tasks/pgmt_perception.py`，验收入口为 `scripts/validate_pgmt_perception.sh`。

| 范围 | 落点 |
| --- | --- |
| glimpse 纯张量模块 | `motion_tracking_rl/networks/pgmt_terrain_glimpse.py` 已实现输入接口、selector、带掩膜 crop、共享 patch encoder、完整独立入口及可选调试副本 |
| IFM / policy metadata | 扩展 `motion_tracking_rl/networks/pgmt_actor_critic.py`；保持 Stage 1 默认分支，显式配置感知及 value groups |
| 权重迁移 | 新增 `motion_tracking_rl/utils/pgmt_checkpoint.py`，配合 runner 明确区分初始化与严格续训 |
| 环境与配置 | 从 `PGMTPlacementEnvCfg` 派生 `pmt_tasks/env_cfgs/pgmt/stage2.py`；新增 `configs/task/pgmt_stage2.yaml`、对应 obs/network 配置；扩展 `pmt_tasks/agent_cfgs/pgmt.py`、`builder.py` 和 `registry_gym.py`，P2 profile 明确使用三头原奖励 |
| 观测噪声 | 扩展 `pmt_tasks/mdp/pgmt_observations.py` / 感知 ObservationGroup；复用 `pgmt_height_scan.py` 的干净契约 |
| 启动与验收 | 新增 `scripts/train_pgmt_stage2.py`、`scripts/validate_pgmt_stage2.sh`、`scripts/report_pgmt_stage2.py` 及纯张量测试；复用现有 Sim 6 启动、Fabric、256 MiB 碰撞栈和 GUI 控件，保留 P1/v5 入口 |

P2 交付物：感知配置与源码、迁移报告、数值/梯度报告、GUI 检查材料、Isaac 短训与续训报告。完成这六项后，进入 P3 的四头奖励与完整训练闭环。

### 7. P2.4 / P2.5 完成记录（2026-10-13）

已按 §4.0 执行。49 项旧状态完整复制、10 项感知状态初始化；实际禁用感知的动作/value 误差均为 0。160 环境完成 25 次 GUI 更新、221 次单次 outcome 结算和两次新进程恢复；L0/L5/L9 的 6,615 个 PhysX 扫描点核对、跨 tile/外地图 truncation 通过。321 项 CPU/CUDA 回归通过。160 三组成本已记录；1024 的初始碰撞显存不足，结果作废，外部错误监视器验证 0 更新/无 checkpoint 退出。详见 [完整报告与命令](PGMT_P245_ACCEPTANCE.zh-CN.md)。所有本轮 Isaac 已关闭；本轮结论不包含地形奖励学习和长期地形适应性能。
