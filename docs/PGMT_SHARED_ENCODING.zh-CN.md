# PGMT actor/critic 同批共享编码

2026-10-15。目标是在不改变观测、奖励、PPO 超参数和模型结构的条件下，消除 actor/critic 对同一批观测的重复编码。

## 实现

- `PGMTActorCritic.act_and_evaluate(obs)` 调用一次 `_encode()`，得到 `proprio` 和 `intent`，随后构建 actor distribution、采样 action，并计算四头 critic value。
- 两个 loss 都通过同一个 `intent` 反传。没有 detach、跨调用 latent 缓存或旧计算图复用；每个 minibatch 重新执行完整编码。
- 网络使用同一份已经增强并保存在 rollout 中的 `terrain_map/terrain_valid`。联合前向不会重复增强，也不会改变 actor/critic 的特权输入边界。
- 保留原 `act()`、`evaluate()`、`act_inference()`。原生 PPO 在 rollout 和 update 中检测联合接口；没有联合接口或使用 recurrent 的策略沿用原来的分开调用。
- PPO 更新时保留原本 `act()` 中的 action sampling，虽然该采样值不参与更新，仍保证 RNG 消耗顺序不变。采集阶段的 mean-action 选项不变。
- 模型张量、normalizer、优化器与训练契约不变，旧 checkpoint 可以严格续训。`policy_metadata` 中历史 `separate forward calls` 描述的是仍然保留的独立 API；实际 PPO 执行方式单独写入 checkpoint 的 `execution_options.actor_critic_forward=joint_encode`，不需要转换原 checkpoint。

代码：

- [联合前向接口](../motion_tracking_rl/networks/pgmt_actor_critic.py)
- [rollout/PPO 更新接入](../motion_tracking_rl/algorithms/ppo.py)
- [执行元数据](../motion_tracking_rl/runners/pgmt_runner.py)
- [新旧路径数值比较](../tests/test_pgmt_joint_forward.py)
- [真实 PPO 梯度与观测保存验证](../tests/test_pgmt_perception_ppo.py)

## 数值与兼容性验证

- pmt CPU：联合接口、感知 PPO、原 PGMT RL 契约测试 **58 passed**。
- pmt CUDA 1：联合接口、感知 PPO、IFM 测试 **68 passed**。覆盖三/四头、有效/混合/全无效地图、禁用感知、scalar/log 与 state-dependent std、随机/mean rollout。
- 同一参数和输入下，action、value、distribution、log probability、entropy、采样 RNG 状态完全一致；共享编码的参数梯度与两次独立编码在数值容差内一致，18 组 CUDA 对照的最大绝对差为 `1.49e-8`。分别用 actor-only、critic-only 和 combined loss 验证编码器梯度。
- 连续两轮四头 PPO 对照验证 rollout、GAE、loss、参数、Adam 状态及 RNG；每个 minibatch 的地图输入、裁剪图和 IFM 输出只生成一份。更新中禁止重新调用地形增强。
- 历史 checkpoint/四头恢复测试在原 NumPy 2 环境 **39 passed**。首次在 pmt / NumPy 1 环境读取旧 P2 文件时有 8 项失败，原因是历史 pickle 引用 `numpy._core`；未修改安装环境、原 checkpoint 或严格恢复检查。

CUDA 测试：[日志](../logs/pgmt_joint_encode_20261015/cuda_tests.log)、[JUnit](../logs/pgmt_joint_encode_20261015/cuda_tests.xml)。恢复测试：[日志](../logs/pgmt_joint_encode_20261015/checkpoint_tests.log)。

## Isaac 性能与续训

已完成 pmt / Isaac Sim 5.1、CUDA 1、8192 环境、headless、完整 v6 地形 A/B。两组从同一 383 checkpoint 恢复，保持 `--performance all`、seed 722、24 控制步、5 epochs × 4 minibatches；各 8 次更新，比较后 6 次。

| 指标 | 原分开编码 | 联合编码 | 变化 |
|---|---:|---:|---:|
| Collection / 轮 | 5.094 ± 0.163 s | 5.050 ± 0.157 s | −0.9%，不足以确认 collection 提速 |
| Returns + PPO / 轮 | 2.856 ± 0.013 s | **1.507 ± 0.018 s** | **−47.2%** |
| Collection + PPO / 轮 | 7.950 s | **6.556 s** | **−17.5%** |
| Torch 峰值 allocated | 15,274.4 MiB | **11,766.0 MiB** | **−23.0%** |
| Torch 峰值 reserved | 16,464 MiB | 13,280 MiB | −19.3% |
| 每轮训练编码次数 | 89 | **45** | 与预期一致 |

调用计数独立检查：原路径 `24×2 + 1 + 20×2 = 89`，联合路径 `24 + 1 + 20 = 45`；其中 `+1` 是 rollout 末尾 bootstrap value。两组每轮另有 1 次日志用 debug 编码，不计入训练次数。

两组 initial 模型、初始物理状态、完整 env/agent 配置逐项相同。27 项奖励及 1078 个 TensorBoard scalar tags 均有限，四头梯度均有限且非零。原路径 1598 次 done、新路径 1592 次 done，各自都与新增课程 outcome 数严格相等。接触/地形/扫描/放置和奖励后检通过。共享反传改变浮点求和顺序，且 PhysX 本身不承诺逐位轨迹复现，因此不要求更新后的物理轨迹逐步相同。

这是共享 GPU 上的顺序短测；上表不承诺长期平均速度。Torch allocated/reserved 不包含全部 PhysX、渲染或其他进程的显存。

完整对照：[comparison.json](../logs/pgmt_joint_encode_20261015/comparison.json)。原数据及 TensorBoard：`logs/pgmt_joint_encode_20261015/cuda1_separate_v2` 与 `cuda1_joint`。

正常入口 `scripts/train_pgmt_stage2_native.sh` 已从新 checkpoint **391→393** 完成 2 次更新，不依赖 benchmark 包装器。模型、normalizer、Adam、课程、sampler 概率、学习率及迭代编号严格恢复；27 项奖励和 1074 个 scalar tags 有限，38 次 done 与新增 outcome 严格匹配。见 [续训报告](../logs/pgmt_joint_encode_20261015/resume_report.json)。

所有测试 worker 已退出，GPU 分配已释放；原正式 383 checkpoint SHA256 不变。见 [最终核验](../logs/pgmt_joint_encode_20261015/final_audit.json)。后续正常 PGMT PPO 训练自动使用联合接口，不需要增加启动参数。

过程与复现入口：[EXECUTION.md](../logs/pgmt_joint_encode_20261015/EXECUTION.md)、[run.sh](../logs/pgmt_joint_encode_20261015/run.sh)。测试目录独立，原正式训练 383 checkpoint 不覆盖。
