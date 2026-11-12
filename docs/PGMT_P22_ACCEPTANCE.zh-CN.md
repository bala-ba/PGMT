# PGMT P2.2 Terrain-Glimpse 独立验收

日期：2026-10-12。**结论：第 1–5 部分已完成，P2.2 独立验收通过，可以进入 P2.3。** 本次验收对象是输入预处理、selector、带掩膜裁剪、共享 patch encoder 和调试输出。IFM、actor/critic、PPO 接入及 Isaac 训练仍按 P2.3–P2.5 执行。

## 1. 交付物与复现

- 验收入口：[validate_pgmt_terrain_glimpse.py](../scripts/validate_pgmt_terrain_glimpse.py)。必须能访问 CUDA；不可用时明确失败，不静默跳过 GPU。
- 网络实现：[pgmt_terrain_glimpse.py](../motion_tracking_rl/networks/pgmt_terrain_glimpse.py)，本次验收没有修改网络计算逻辑。
- 最终报告：[report.json](../logs/pgmt_p22_acceptance/full_20261012_165716/report.json)，包含配置、源码/输入哈希、误差、梯度、中心分布和全部性能数据。
- 回归结果：[pytest.log](../logs/pgmt_p22_acceptance/full_20261012_165716/pytest.log)、[pytest.xml](../logs/pgmt_p22_acceptance/full_20261012_165716/pytest.xml)。
- 实际地图调试数据：[real_maps_debug.npz](../logs/pgmt_p22_acceptance/full_20261012_165716/real_maps_debug.npz)；[中心与 patch 覆盖图](../logs/pgmt_p22_acceptance/full_20261012_165716/real_maps_glimpses.png)。图中灰色为无效扫描点，彩色方框表示 0.4 m 采样跨度，坐标为实际 torso yaw 系。

在能访问 GPU 的终端执行：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_terrain_glimpse.py \
  --device cuda:0 --seed 2245 --batches 1 160 1024 8192 \
  --warmup 10 --iterations 30 --cuda-inner-steps 10
```

默认读取 `logs/pgmt_p21_acceptance/20261012_pmt_alignment_v3_r1/` 的 version=3 契约及五类地形 L0/L5/L9、L9 增强地图，共 20 张。输出自动写入 `logs/pgmt_p22_acceptance/full_<时间>/`，不覆盖已有验收。脚本检查米制、高度符号和归一化位置，干净地图还核对已有 PhysX 对照值；随机条件和增强压力样例在验收端固定采样，网络内部不重新加噪。

## 2. 环境与固定配置

| 项目 | 本次配置 |
| --- | --- |
| GPU | `cuda:0`，NVIDIA RTX 5090，32,607 MiB；驱动 595.58.03 |
| PyTorch | 2.7.0+cu128；CPU 1 线程；float32 matmul=`highest`，关闭 TF32 |
| 预处理 | 米制地图 `clip(-3,3)/3`，每次完整 forward 执行一次；PMT 符号/无效值保持原契约 |
| selector | `1498→256→128→8`，ELU，中心 `0.8×tanh(raw)` |
| 裁剪 | 4 个 5×5 patch，间隔 0.1 m；双线性有效权重归一化；点权重 ≥0.5；至少 13/25 点有效 |
| encoder | 四个 patch 共用 `52→128→128` MLP，ELU 隐藏层，线性输出；整个模块 440,968 参数 |
| 输出 | `terrain_tokens[N,4,128]`、bool `token_valid[N,4]`；可选独立调试副本 |

## 3. 功能、数值和梯度

相关回归一次完整运行 **157 passed，0 failed，0 skipped**。范围为 `test_pgmt_terrain_glimpse.py`、`test_pgmt_perception.py`、`test_pgmt_rl.py`、`test_normalization_resume.py`，包含四项 CUDA 前向/反向检查。

| 验收项 | 结果 |
| --- | --- |
| 解析几何 | 常数平面、X/Y 斜坡、台阶、非整数中心、四个 yaw 的轴序，以及 N=1/批量均通过 |
| 缺测与边界 | 无效填充值/NaN 不污染有效插值；权重 0.5、有效点 12/25 与 13/25 边界、扫描越界和整图无效均通过 |
| 共享编码 | 52 维特征顺序、patch 重排、单组共享权重及配置/state_dict 严格重载通过 |
| 调试与确定性 | 同设备重复前向逐值相同；不消耗 RNG、不修改输入；调试副本无图/无别名，开关不改变输出或梯度 |
| 梯度路径 | 去掉 encoder 的两个中心坐标输入贡献后，token 损失仍经裁剪高度回传 selector；CPU/CUDA 的全部参数及地图/history/commands 梯度有限且非零 |
| 有限差分 | 含缺测邻点、包含插值分母导数的 double 检查：CPU/CUDA 最大绝对误差均约 `4.08e-11` |

实际地图每种输入状态分别检查 selector 中心、四角中心、扫描范围外中心。5 种输入状态 × 20 张地图 × 3 组中心 × 4 个 patch × 25 点 × 2 个设备，共 **60,000 次采样点对照**；NumPy oracle 独立按邻格权重逐点计算，不调用 `grid_sample`。

| 数值指标 | 最大绝对误差 |
| --- | --- |
| 与 NumPy 对照：归一化裁剪高度 | `2.55999e-7`，对应米制约 `7.68e-7 m` |
| 与 NumPy 对照：有效插值权重 | `2.83228e-6` |
| CPU/CUDA：terrain token | `2.38419e-7` |
| CPU/CUDA：中心坐标 | `1.63913e-7 m` |
| CPU/CUDA：裁剪高度 | `3.57628e-7` |
| CPU/CUDA：参数及输入梯度 | `1.86265e-9` |

逐点掩膜与 patch/token 掩膜在 oracle 对照及 CPU/CUDA 对照中均完全一致。这里的高度误差衡量对已有扫描网格的双线性插值实现，不代表对未扫描位置的真实碰撞面测量误差。

## 4. 无效输入与中心分布

下表为种子 2245、随机初始化、零 history/已归一化 command 条件下，selector 实际输出位置的统计。每行 20 张地图、80 个 patch；CPU/CUDA 有效标记一致。

| 输入状态 | 有效采样点比例 | 有效 patch 比例 |
| --- | ---: | ---: |
| 原保存地图（包含 L9 增强图） | 98.05% | 100%（80/80） |
| 在原掩膜上额外丢失 10% 点 | 90.15% | 100%（80/80） |
| 中央 7×7 点缺测（中心跨度 0.6 m） | 5.15% | 0% |
| 屏蔽 x>0 的地图区域 | 54.95% | 73.75%（59/80） |
| 全图无效 | 0% | 0% |

上述所有状态均输出有限 token。中央孔洞/全无效时不会重新选点，`token_valid=False` 会保留下来；P2.3 必须将它用于 attention logits 屏蔽。100% 保留率仅为本次样例统计，不能解释为任意缺测情况下均有效。

另用 5 个初始化种子（2245–2249），每个种子分别输入 1,024 组零条件和 1,024 组标准正态合成 history/commands，共 **10,240 组环境输入、40,960 个中心**。地图为上述 20 张图重复取样；合成条件不是实际策略 rollout。

| 中心指标 | 零条件 | 标准正态合成条件 |
| --- | ---: | ---: |
| 全部坐标范围 | `[−0.1773, 0.2659] m` | `[−0.3815, 0.4548] m` |
| 接近边界比例（单坐标绝对值 ≥0.79 m） | 0% | 0% |
| 中心对距离 <0.1 m 的比例，跨种子范围 | 4.15%–61.69% | 6.20%–28.30% |
| patch 面积重叠比例，跨种子范围 | 100% | 89.06%–99.54% |
| 平均两两 IoU，跨种子范围 | 0.221–0.573 | 0.190–0.420 |

**未训练中心偏向扫描中央并有明显重叠，未发生边界饱和。** 共享 MLP 不要求四个 patch 分离；本次没有新增多样性损失或强制分散规则。P2.3/P2.5 接入训练后继续观察重叠、覆盖率和中心变化，独立验收不能证明选点性能。

## 5. 批量耗时与显存

每种模式预热 10 次，记录 30 组：CPU 每组一次，CUDA 每组连续 10 次并在组末同步，报告每次调用的墙钟时间中位数。CUDA Event 的时间跨度及 P95 也保存到 JSON；Event 包含 host dispatch 间隙，不能视为纯 kernel 时间。首个分配器冷启动测量组保留在统计内。

| Batch | CPU 前向 ms | CPU 前向+反向 ms | GPU 前向 ms | GPU 开启调试前向 ms | GPU 前向+反向 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.147 | 0.544 | 0.294 | 0.419 | 1.076 |
| 160 | 1.915 | 5.118 | 0.356 | 0.360 | 1.191 |
| 1,024 | 12.204 | 31.931 | 0.418 | 0.422 | 1.213 |
| 8,192 | 106.728 | 290.056 | 0.814 | 0.825 | 1.687 |

| Batch | GPU 前向 P95 ms | GPU 前向+反向 P95 ms | 前向峰值 allocated MiB | 前向+反向峰值 allocated MiB | 前向+反向额外峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.507 | 1.269 | 18.95 | 20.46 | 2.52 |
| 160 | 0.449 | 1.272 | 21.33 | 23.70 | 5.06 |
| 1,024 | 0.449 | 1.313 | 37.83 | 53.20 | 30.14 |
| 8,192 | 1.664 | 3.341 | 175.13 | 297.19 | 242.03 |

峰值使用 PyTorch `max_memory_allocated`，包含模型参数、输入及库工作区；“额外峰值”减去测量开始前的 live allocation。数值不包含驱动/context 显存、优化器、Isaac、扫描器、IFM、PPO storage 或旧策略主干。反向包括 history 输入梯度；没有 optimizer step。输入使用重复的已保存地图和零条件，不重新采样增强。以上为独立模块成本，完整训练吞吐和总显存由 P2.5 测量。

## 6. 发现与处理

1. **GPU 环境隔离：** 前序测试沙箱缺少 `/dev/nvidia*`。本次在可访问设备的执行环境完成验收，并让验收入口在 CUDA 不可见时直接报错；不再把跳过检查作为完成状态。
2. **短调用计时波动：** [首轮报告](../logs/pgmt_p22_acceptance/full_20261012_165442/report.json) 的功能检查通过，但单次 GPU 调用受同步/调度影响明显。本次改为每组 10 次调用，同时记录墙钟、CUDA Event 和 P95；上表采用最终分组计时报告。即使如此，大 batch P95 仍高于中位值，保留原始数据，不将单个中位值视为稳定训练速度保证。
3. **未训练中心重叠：** 已量化并保存，作为后续学习过程的诊断基线；不改动当前网络结构。掩膜通过不等于已完成 IFM 屏蔽，后者仍属 P2.3。

本次交付了可重复执行的完整独立验收、数据和可视化，未启动新的 Isaac 仿真或 PPO 训练。下一步为 P2.3 的 terrain K/V 接入、无效 token 屏蔽及 actor/critic/PPO 梯度验证。
