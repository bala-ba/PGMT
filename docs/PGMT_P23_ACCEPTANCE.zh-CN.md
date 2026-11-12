# PGMT P2.3 CPU/CUDA 融合验收

日期：2026-10-13。**P2.3 第 1–5 部分完成，融合验收通过。** 本次使用完整 actor/critic 网络、真实保存地图与固定权重夹具，验证 Terrain-Glimpse → IFM → actor/critic → 原生 PPO 的连接。无需修改网络、PPO 或 RolloutStorage。P2.4 的指定 checkpoint 迁移/严格续训、P2.5 的 Isaac 训练仍待实施。

## 1. 交付物与复现

- 独立入口：[validate_pgmt_ifm.py](../scripts/validate_pgmt_ifm.py)。要求可见 CUDA，不静默跳过 GPU；每次创建独立输出目录。
- 最终结果：[report.json](../logs/pgmt_p23_acceptance/fusion_20261013_144840/report.json)，包含源码/地图哈希、旧权重复制清单、配置、逐地图诊断、误差和全部计时/显存样本。
- 回归：[pytest.log](../logs/pgmt_p23_acceptance/fusion_20261013_144840/pytest.log)、[JUnit](../logs/pgmt_p23_acceptance/fusion_20261013_144840/pytest.xml)。**285 passed，0 failed，0 skipped**；其中第四部分的原生 PPO 用例包含 240 次真实 Adam 更新。
- 独立融合检查另覆盖 CPU/CUDA 各 100 个真实地图变体、100 次 N=1 检查和 N=100 混合掩膜批量；性能测量额外完成 **384 次 Adam 更新**，本次完整执行合计 624 次。
- 调试数据：[fusion_debug.npz](../logs/pgmt_p23_acceptance/fusion_20261013_144840/fusion_debug.npz)；[地图、中心与逐 head 地形注意力图](../logs/pgmt_p23_acceptance/fusion_20261013_144840/fusion_real_maps.png)。图为未训练权重夹具，4 个中心尚有重叠，不代表已经学会选择落脚区域。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_ifm.py \
  --device cuda:0 --seed 23505 --batches 160 1024 \
  --warmup 5 --iterations 20 --ppo-iterations 5
```

默认复用 P2.1 `20261012_pmt_alignment_v3_r1` 中 flat/stairs/slope/boxes/rough 的 L0/L5/L9 与 L9 增强图，共 20 张。校验 version=3、米制、sensor−ground 符号、网络侧归一化；干净图再次比对已有 PhysX 高度数组。proprio/history/commands/privileged 是固定种子的合成输入；**真实数据仅指地形扫描，不是完整机器人轨迹**。

## 2. 固定配置与覆盖

| 项目 | 配置 |
| --- | --- |
| 硬件/数值 | RTX 5090 `cuda:0`，驱动 595.58.03；PyTorch 2.7.0+cu128；CPU 1 线程；float32、关闭 TF32，无 autocast/compile |
| actor / critic | 隐藏层均为 `[512,256,128]`；输入 224 / 379；29 维动作均值与 upper/lower/auxiliary 三头 value |
| IFM | 128 维、4 heads；9 个旧 token 加 4 个地形 token；既有 RoPE 与逐环境 key 掩膜 |
| 参数 | 整网 1,235,781；新增 Terrain-Glimpse 440,968 |
| 旧状态复制 | 先用不同种子构造旧/新实例，再显式复制 49 个旧 state_dict 项；只允许新增 10 个 `terrain_glimpse.*` 项；逐项检查名称/shape/dtype，合并完整字典后 `strict=True` 加载并逐值复核 |
| normalizer | 一次校准后固定统计；actor/critic count=80，command count=640。12 个归一化 buffer 全部包含在复制核验中 |
| 地图状态 | 原样、额外 10% 丢点、中央缺测、+X 半图越界、全无效；20×5=100 个样例，保存后网络不重新采样 |

旧行为对照使用显式的原 9-key 编码路径，分别比较 Stage-1 实例、新模块已构建但 disabled 的实例，以及 enabled 但所有地形无效的实例。这个复制函数是**验收夹具**，没有加载 `model_15000.pt`，不替代 P2.4 正式迁移接口。

## 3. 功能与数值结果

| 检查 | 结果 |
| --- | --- |
| disabled → 旧网络 | CPU/CUDA 的 intent、动作均值、三头 value 全部逐值一致，误差 0 |
| 全无效 terrain → 旧网络 | CPU 最大误差 `5.96e-8`，CUDA 误差 0；均满足 `atol=rtol=1e-5` |
| N=1 → 混合批量 | 各设备逐张核对全部 100 个样例，最大输出误差 `1.79e-7`，key 掩膜完全相同 |
| 正常/调试路径 | intent、动作均值、value 逐值一致；公开调试入口的 intent 一致；输入、RNG 和全部 normalizer buffer 不变 |
| 屏蔽规则 | 每个 head 的无效 key 权重精确为 0，权重和为 1；所有地图全无效时四个 terrain token 的注意力占比为 0 |
| 无效 token 扰动 | 在 IFM 输入处将被屏蔽的 K/V token 改为 71，三个输出均逐值不变；没有将此断言误用于 selector 读取的原始无效高度 |
| 有效地图响应 | 固定其余输入，仅将有效高度增加 0.3 m（保留掩膜、按契约裁剪），20 张源图的 intent、动作均值、value 全部改变；最小的单图最大变化分别约 `5.55e-3 / 2.90e-4 / 1.46e-4` |
| NumPy 裁剪对照 | CPU/CUDA 各 10,000 个点；归一化高度最大误差 `2.43e-7`；点掩膜与 patch 掩膜完全一致 |
| PPO 真实更新 | 第四部分回归再次通过；完整网络、真实地图基准下 enabled 的 selector/encoder/IFM 梯度均有限且非零，disabled 的感知梯度为零；所有 loss 与受检参数有限 |

完整 enabled 网络的 CPU/CUDA 最大绝对误差如下；全部通过 `atol=rtol=1e-5`，bool 掩膜要求精确一致。

| 项目 | 最大误差 |
| --- | --- |
| intent | `8.94e-8` |
| 动作均值 | `1.42e-7` |
| 三头 value | `1.49e-7` |
| 中心坐标 | `2.20e-7 m` |
| 归一化 patch 高度 | `3.87e-7` |
| 插值有效权重 | `2.86e-6` |
| 逐 head attention | `2.98e-8` |
| point / patch / key 掩膜 | 全部一致 |

这些结果证明融合、屏蔽、存储重算与可训练性，不表示当前未训练 selector 已具有地形适应能力。

## 4. 整网成本

**GPU 耗时受并行负载影响。** 验收开始前 `nvidia-smi` 已显示 GPU 0/1 利用率约 95%/94%，结束时约 99%/91%；原始信息保留在 JSON 中。本次未停止其他 GPU 工作。以下为当时共享负载下的实测值，不能用来推导独占加速比或 Isaac 吞吐。

前向预热 5 次、测量 20 次，均为 `inference_mode`，分别计时 `act_inference()`、`evaluate()` 及顺序调用两者。每次 CUDA 测量同步，记录 wall time 与 CUDA Event；Event 区间也包含主机提交空隙，不解释为纯 kernel 耗时。

PPO 预热 1 轮、测量 5 轮，每轮 **4 步 rollout、2 epochs×4 minibatches，共 8 次 Adam step**，minibatch 大小等于表中 batch。计时包含原生 Gaussian act/value 重算、loss、backward、梯度裁剪与 Adam，排除 rollout/GAE 准备及验收断言。学习率固定 `1e-4`。两种模式均构建完整感知模型并保存地图，disabled 跳过感知计算，用于对照计算开销。

单位 ms；前三项为中位数，PPO 列为中位数 / P95。

| 设备 | batch | 感知 | actor | critic | actor+critic | PPO 更新 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 160 | disabled | 3.55 | 3.76 | 7.33 | 216.22 / 217.40 |
| CPU | 160 | enabled | 6.10 | 6.38 | 12.80 | 343.73 / 367.53 |
| CPU | 1024 | disabled | 24.01 | 25.37 | 49.11 | 1218.61 / 1227.11 |
| CPU | 1024 | enabled | 40.70 | 42.30 | 83.73 | 1937.87 / 1948.56 |
| RTX 5090 | 160 | disabled | 2.20 | 2.34 | 1.78 | 196.23 / 204.96 |
| RTX 5090 | 160 | enabled | 2.52 | 1.11 | 2.92 | 237.90 / 246.85 |
| RTX 5090 | 1024 | disabled | 2.51 | 0.67 | 3.05 | 161.70 / 165.50 |
| RTX 5090 | 1024 | enabled | 2.81 | 1.15 | 3.52 | 214.43 / 225.78 |

各 API 在不同时段独立测量，GPU 并行负载有波动，表中存在两者合计短于某个单独 API、1024 比 160 的 PPO 更新更快等现象；保留原始数据，不据此推断规模扩展规律。P2.5 再在匹配负载下测实际采样、更新和渲染成本。

CUDA 内存单位 MiB，记录 PyTorch allocated 峰值；**不是整机 `nvidia-smi` 占用**。前向包含当时进程内模型/输入/工作区等 live allocations；PPO 还包含 rollout、梯度与已预热的 Adam 状态。额外峰值减去该次调用前的 allocated 基线，reserved 另存 JSON。

| batch | 感知 | actor+critic 峰值 | PPO 峰值 | PPO 调用额外峰值 |
| --- | --- | --- | --- | --- |
| 160 | disabled | 19.18 | 63.60 | 28.11 |
| 160 | enabled | 29.18 | 76.99 | 38.13 |
| 1024 | disabled | 62.34 | 256.49 | 174.39 |
| 1024 | enabled | 74.67 | 318.92 | 233.47 |

这不是不构建感知模块的原始 Stage-1 内存基线；也未包含 Isaac、扫描器、PhysX 和 GUI。不能将这些数字直接当作 8192 环境正式训练显存预算。

## 5. 状态与后续边界

没有发现需要修改策略、原生 PPO 或 storage 的数值/梯度问题。本次新增独立验收入口并保存完整证据；性能测量中的并行 GPU 负载作为限制明确记录，未改变系统中其他任务。

**P2.3 完成，可进入 P2.4。** 下一阶段按计划提供正式任务配置、指定 `model_15000.pt` 的逐项迁移、保存与严格续训入口；P2.5 再执行 Isaac 可视化、真实训练和恢复检查。本次未启动 Isaac、未替换训练 checkpoint，也未修改奖励或课程逻辑。
