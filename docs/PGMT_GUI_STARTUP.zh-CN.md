# PGMT Stage2 可视化启动排查

日期：2026-10-15。当前机器可视化使用既有 `scripts/train_pgmt_terrain.sh`（Isaac Sim 6.0.1、GPU 1）；正式无界面训练使用 `scripts/train_pgmt_stage2_native.sh`（pmt / Isaac Sim 5.1）。

## 故障与结论

- 用户运行 `logs/pgmt_stage2_center2x2/gui_01` 时，在 `pmt` 环境直接以 `--device cuda:0` 开启 GUI。Kit 日志报 `Failed to find a graphics and/or presenting queue`、`createSwapchain failed`，随后原生崩溃，尚未进入环境或 PPO 初始化。显示器连接在物理 GPU 1，GPU 0 的 `display_active` 为 Disabled。
- 改用 GPU 1 进行有限复测后，呈现队列报错消失，但 Sim 5.1 仍在约 8 秒、`app ready` 后的界面初始化中发生原生崩溃。仅切换 GPU 不足以解决。本次记录：[控制台](../logs/pgmt_gui_fix_20261015/native_cuda1_01.console.log)。该进程卡在 crash reporter 后已清理，未进行训练。
- 这与 [P1 历史记录](PGMT_P1_ACCEPTANCE.zh-CN.md) 中 Sim 5 GUI 的失败一致。沿用项目已实现的 Sim 6 兼容入口，不修改安装环境、训练公式或 checkpoint。之前给出的 `pmt + GUI + cuda:0` 命令不适用于当前机器。
- 原生崩溃发生在 Python 主逻辑之前，失败目录的 `training_status.json` 可能停留于 `launching`；它不代表进程仍在运行。本次没有改写历史失败状态。

## 实际验证

运行目录：[sim6_cuda1_01](../logs/pgmt_gui_fix_20261015/sim6_cuda1_01)。使用当前中央 2×2 随机出生区域、五类地形、完整感知与四头 critic，从 Stage1 `model_15000.pt` 迁移，160 环境、`headless=False`、`--performance all`。

- **2 次 PPO 更新、48 个控制步、7680 个环境步**完成，进程正常退出，状态为 `completed`，`final.pt` 已保存。
- 四头损失均有限；奖励分组求和最大误差 `1.1921e-7`；源 Stage1 checkpoint 哈希保持不变。
- 已查看 [训练截图](../logs/pgmt_gui_fix_20261015/sim6_cuda1_01/training_viewport_1.png)，机器人、有效/无效扫描点、glimpse 中心及边框正常显示。全景和固定楼梯 tile 截图同目录保存。
- 记录：[状态](../logs/pgmt_gui_fix_20261015/sim6_cuda1_01/training_status.json)、[逐轮日志](../logs/pgmt_gui_fix_20261015/sim6_cuda1_01/progress.jsonl)、[控制台](../logs/pgmt_gui_fix_20261015/sim6_cuda1_01.console.log)。这是 GUI 启动和短训验证，不是完整训练收敛验收。

## 可视化命令

无需手动激活 conda；脚本使用绝对解释器路径，并配置兼容层与 GPU 1。每次使用新日志目录，避免覆盖。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
HEADLESS=0 LIVESTREAM=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
bash scripts/train_pgmt_terrain.sh \
  --num_envs 160 --max_iterations 1000 --save_interval 50 \
  --performance all \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir "logs/pgmt_stage2_center2x2/gui_$(date +%Y%m%d_%H%M%S)"
```

该命令执行带可视化的训练。界面保留全部机器人/自由相机、单机器人跟随、指定地形与等级的固定 tile 视角。大规模训练继续使用 native headless 入口；两种入口的仿真版本不同，不将这次 GUI 测试解释为跨版本严格续训验证。
