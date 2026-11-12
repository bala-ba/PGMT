# PGMT Stage2 单 GPU / 多 GPU 训练

`scripts/train_pgmt_stage2_native.sh` 支持单机多 GPU。每张 GPU 运行独立的 Isaac Sim 环境进程，共同训练同一套策略；PPO 梯度、KL 学习率判断和观测归一化统计跨卡同步。

## 双 GPU 启动

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --gpus 0,1 \
  --num_envs 8192 \
  --max_iterations 20000 \
  --save_interval 100 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir logs/pgmt_stage2_center2x2/train_2gpu_01
```

`--num_envs` 始终表示全部 GPU 合计的环境数。上例每张卡 4096 个环境，每次 PPO 更新合计采集 8192 个环境的 rollout；如需每张卡 8192 个环境，传入 `--num_envs 16384`。总数必须能被 GPU 数整除，且每张卡至少 5 个环境。各 rank 使用 `seed + rank`，课程和动作采样器各自维护。

`--gpus` 设置本次子进程的 `CUDA_VISIBLE_DEVICES`，并通过当前 Python 环境的 `torch.distributed.run` 启动；不需要单独运行 torchrun。指定 GPU 在进程内重新编号为 `cuda:0`、`cuda:1` 等，`--gpus` 优先于 `--device`。多 GPU 入口要求 headless 感知训练。

当前机器的 `pmt` 环境自带 NCCL 2.26.2，在双 RTX 5090 上复现了通信初始化阻塞/非法访存；新版 NCCL 2.27.5 已通过双卡测试。native 脚本默认仅为多 GPU 子进程预加载已有 `isaacsim` 环境的 `libnccl.so.2`，不修改 Conda 包。迁移机器时可用 `PGMT_NCCL_LIBRARY=/path/to/libnccl.so.2` 指定兼容版本；若当前环境已兼容，设置 `PGMT_NCCL_LIBRARY=` 可禁用覆盖。相关上游问题见 [PyTorch #152780](https://github.com/pytorch/pytorch/issues/152780)。

## 单 GPU 启动

原有参数不变：

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --device cuda:1 \
  --num_envs 8192 \
  --max_iterations 20000 \
  --save_interval 100 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir logs/pgmt_stage2_center2x2/train_single_02
```

单 GPU 不创建分布式通信组。省略 `--gpus` 时，`--device` 遵循调用终端现有的 CUDA 可见设备编号。

## 保存、恢复与停止

只有 rank 0 写 TensorBoard、`progress.jsonl` 和模型文件。`model_*.pt`、`initial.pt`、`final.pt` 各自包含共享模型/优化器，以及所有 rank 的课程、动作采样器、RNG 和训练契约，无需搬运额外检查点文件。其他 rank 的配置、诊断和状态位于 `ranks/rank_<N>/`。

TensorBoard 中 PPO 损失是跨卡均值，采样步数为全局总数；环境奖励、课程、感知和显存诊断显示 rank 0 的数据。

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --gpus 0,1 --num_envs 8192 \
  --max_iterations 20000 --save_interval 100 \
  --resume logs/pgmt_stage2_center2x2/train_2gpu_01/final.pt \
  --log_dir logs/pgmt_stage2_center2x2/resume_2gpu_01
```

严格恢复要求保持 GPU 数量、各 rank 的环境数、种子和训练配置一致。单卡检查点不能作为双卡的 `--resume`，双卡检查点也不能按单卡严格恢复；Stage1 的 `--initialize-from` 可用于启动单卡或双卡新训练。物理 episode 会重新开始。

在训练目录创建 `STOP_REQUESTED`，所有 rank 会在同一次 PPO 更新完成后停止，并共同保存 `final.pt`：

```bash
touch logs/pgmt_stage2_center2x2/train_2gpu_01/STOP_REQUESTED
```

`--max_runtime_s` 或任一 worker 收到退出信号时，也会协调停止。原生 PhysX 错误会终止本次训练进程组并标记失败；torchrun 不自动重启失败的 worker。每次启动使用新的日志目录。

## 验证

```bash
LD_PRELOAD=/home/lzp/miniconda3/envs/isaacsim/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_distributed.py tests/test_pgmt_process_guard.py \
  tests/test_pgmt_checkpoint.py tests/test_normalization_resume.py
```

分布式回归实际启动两个进程，验证 PPO 参数/Adam 同步、全局归一化、完整检查点恢复，以及非主 rank 发起的停止。无 GPU 时运行 Gloo CPU 测试；两张 CUDA GPU 可用时额外运行 NCCL 测试。

2026-10-15 验证：上述测试加感知 PPO、共享编码与 PPO 日志回归合计 **107 项通过**；NumPy 2 环境的四头检查点专项 **25 项通过、2 项 CUDA 测试跳过**。真实 Isaac Sim 验证包括单卡 10 环境 1 次更新、双卡合计 20 环境 2 次更新、双卡严格恢复后追加 1 次更新并按时间限制共同停止，以及默认 `--performance all` 的双卡恢复更新。结果分别位于 `logs/pgmt_multi_gpu_validation/{single,dual_v2,dual_resume,dual_default}/`。这些是功能验证，不代表 8192 环境性能或长期收敛测试。
