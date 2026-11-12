# PPO 日志标量集中读取

2026-10-15。在 actor/critic 联合编码基础上，减少 PPO 常规日志的 GPU→CPU 同步，不改变训练目标或超参数。

## 改动

- 新增 [PPOLogAccumulator](../motion_tracking_rl/algorithms/ppo_logging.py)：每个 minibatch 将日志标量 detach、打包，在设备上累计；整轮结束后将所有分组一次 `.cpu().tolist()`，返回原来的 `dict[str, float]`。
- [PPO.update](../motion_tracking_rl/algorithms/ppo.py) 接入 value/surrogate/entropy/sonic、各 value head、ratio/KL/clip 统计，以及可选辅助损失、RND、symmetry 指标。原日志键、全局 minibatch 平均分母及缺失可选项按零贡献处理的规则不变。
- 常规标量用 float64 累计，匹配旧 Python float 累计精度；各 value head 保留 float32 累计与除法。缓存中只有独立的 detached 小向量，不保留 minibatch 计算图，也不与模型缓冲区共用存储。
- KL 自适应学习率、异常数值检查、显式 debug 输出仍按原时机执行。此次不推迟影响训练决策的读取，也不关闭异常检查。
- `_compute_policy_aux_loss(..., defer_stats=True)` 在 PPO 中返回 detached 日志张量；默认仍返回原来的 Python float，兼容 ADD_PPO 等既有调用者。外部 policy getter 已返回的 Python 数值按原方式累计。
- checkpoint 增加执行元数据 `execution_options.ppo_log_scalars=gpu_accumulate_single_read`，模型/优化器/训练契约不变，无需新启动参数。

当前 PGMT 每轮 20 个 minibatch：9 个常规标量 × 20 次读取，加上轮末一次四头 value 日志读取，由 **181 次读取降为 1 次打包读取**。这不包括保留的 KL 判断、异常检查和日志以外的同步。

## 数值验收

- pmt CPU 回归：**61 passed**；CUDA 1 回归：**37 passed**；旧 checkpoint/四头恢复回归：**39 passed**。
- 对照使用保存的修改前完整 `PPO.update()` 和辅助损失方法，同一 rollout、模型、RNG；覆盖单/三/四头、fixed/adaptive 学习率、辅助损失、Tensor/float policy 统计、连续两轮更新。
- CPU 8 组对照日志误差为 **0**；CUDA 8 组最大绝对差约 **7.11e-15**。每组梯度、参数、Adam 状态、RNG 和每个 minibatch 的学习率一致。
- 累计器检查验证：minibatch 内没有标量读取，轮末只有一次 host transfer；原 tensor 被后续修改时日志快照不受影响；不丢弃 NaN/Inf；空指标、动态出现的指标及混合 Python/Tensor 标量均保持平均语义。

脚本与原始结果：[validate_numeric.py](../logs/pgmt_ppo_scalars_20261015/validate_numeric.py)、[CPU](../logs/pgmt_ppo_scalars_20261015/numeric_cpu.json)、[CUDA](../logs/pgmt_ppo_scalars_20261015/numeric_cuda.json)。

## Isaac 性能与续训

已完成 CUDA 1 / 8192 环境有限 A/B；两组均保留联合编码，使用相同 383 checkpoint、seed 722、完整 v6 地形、headless、`--performance all`。每组 8 次更新，前 2 次热身不计入均值。原正式 checkpoint 不覆盖。

| 指标 | 逐项读取 | 集中读取 |
|---|---:|---:|
| 常规日志读取 / 每轮 | 181 次 | **1 次** |
| Collection / 轮 | 5.053 ± 0.136 s | 5.056 ± 0.147 s |
| Returns + PPO / 轮 | 1.522 ± 0.006 s | 1.506 ± 0.027 s |
| Collection + PPO / 轮 | 6.576 s | 6.563 s |
| Torch 峰值 allocated | 11,762.8 MiB | 11,763.5 MiB |

同步次数的减少已确认；本次 PPO 均值下降约 1.06%，合计仅下降 0.20%，小于共享 GPU 短测的波动量级，**不能认定有稳定的整体提速**。每个 minibatch 的即时非有限值检查和 KL 决策依然保留。

两组初始模型、物理状态、env/agent 配置逐项相同；27 项奖励、1078 个 TensorBoard scalar tags 有限。四头梯度正常，done/outcome 分别严格匹配（1635 / 1622）；现有地形、扫描、接触、放置和奖励后检全部通过。结果：[comparison.json](../logs/pgmt_ppo_scalars_20261015/comparison.json)。

正常 native 入口已从集中读取组的 checkpoint **391→393** 完成 2 次 PPO：模型、normalizer、Adam、课程、sampler 概率和学习率严格恢复；27 项奖励、1074 个 scalar tags 有限，38 次 done/outcome 一致。见 [续训报告](../logs/pgmt_ppo_scalars_20261015/resume_report.json)。

所有测试 worker 已退出，原正式 383 checkpoint SHA256 未变，测试后的源码哈希与验收版本一致。见 [最终核验](../logs/pgmt_ppo_scalars_20261015/final_audit.json)。正常训练自动启用集中读取；过程见 [EXECUTION.md](../logs/pgmt_ppo_scalars_20261015/EXECUTION.md)。
