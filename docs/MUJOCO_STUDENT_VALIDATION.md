# PPOft student 独立 MuJoCo 验证

2026-09-23：`ppoft_vision_bigmap_walkdance.pt` 已在 CPU MuJoCo 3.3.5 中完成独立闭环运行。三组 Bigmap 放置各运行 29.98 秒，均未跌倒、重置或出现求解器警告；位置跟踪存在漂移，未通过本次设置的严格跟踪标准。

## 测试结果

| 配对 motion 索引 | 仿真时长 | 控制步数 / mj_step 次数 | 平均身体位置误差 | 最大身体位置误差 | 跌倒 |
|---|---:|---:|---:|---:|---|
| 0 | 29.98 s | 1499 / 5996 | 0.303 m | 0.715 m | 无 |
| 49 | 29.98 s | 1499 / 5996 | 0.350 m | 0.959 m | 无 |
| 98 | 29.98 s | 1499 / 5996 | 0.328 m | 0.922 m | 无 |

身体位置误差为每一步 14 个 tracked body 的世界坐标距离均值，参照该 raw clip 配对的 optimized trajectory。表中再对时间求平均或最大值。三个配置对应同一 walk/dance 源动作的不同地形放置，不代表三个独立的人类动作类别，也不能据此推断所有动作和地形上的泛化能力。

本次跟踪标准是：全程每一步上述误差小于 0.5 m，无跌倒（pelvis 世界高度低于 0.30 m，或其局部上轴的世界 z 分量小于 0.2），且无求解器警告。此标准是本次评估阈值，不是论文原始成功率定义。三组均满足独立运行和保持站立的检查，但 `tracking_passed=false`。尚未做同初态、同参考帧、同扰动的完整 Isaac 对照实验，不能把这些误差全部归因于更换物理引擎。

结果与视频：

- [汇总 JSON](../logs/sim2sim/summary.json)
- [位置误差和 pelvis 高度曲线](../logs/sim2sim/tracking_curves.png)
- [30 秒独立 MuJoCo 运行视频](../logs/sim2sim/student_motion0_video/testing.mp4)
- [视频验证记录](../logs/sim2sim/student_motion0_video/verification.json)
- 各次运行的 `logs/sim2sim/student_motion{0,49,98}/metrics.json`、`states.jsonl`、`metadata.json` 和 `model/scene.xml`。

## 独立运行的实现和数值校验

运行入口是 `scripts/mujoco_eval_student.py`，适配器位于 `pmt_tasks/backends/mujoco_standalone.py`。评估过程不导入 Isaac Lab／Omniverse，不启动 Isaac Sim；状态由 `mujoco.mj_step` 推进，高程由 `mujoco.mj_ray` 查询，策略每 20 ms 重新推理一次。总计执行 4497 次控制更新和 17988 次 MuJoCo 物理步进。没有逐帧覆写机器人姿态，没有用参考动作代替策略控制。

采用与训练相同的 G1 URDF，按 Isaac 配置将碰撞圆柱替换为胶囊，使用导出的 PD 增益、armature、力矩限制和默认关节角。Bigmap STL 分解为 2499 个独立方块加地面，保留台阶及间隙，避免整个凹形地形被 MuJoCo 当作一个凸包碰撞。关节、刚体均按名字映射。

- 物理步长 0.005 s，decimation=4，策略频率 50 Hz。
- student 控制约定：`q_target = raw_q_ref + action`，scale=1；原有 checkpoint 未修改，也没有微调。
- 历史观测长度 10，参考窗口 21，17 × 11 高程网格与有效性掩码。
- 保留默认高程噪声 ±0.06 m 和 10% dropout；NumPy 随机流与 Isaac 的 PyTorch 随机流不同。
- 每次从配对 optimized clip 的第 0 帧姿态及速度初始化，不施加随机初态扰动，不自动重置。
- Actor 实际使用的所有观测均重建；critic 特权观测不参与推理。

`scripts/export_mujoco_contract.py` 只在准备阶段运行 Isaac，导出静态任务参数和 20 个数值基准样本至 `logs/sim2sim/contract.pt`。随后独立程序先比较同状态的观测、动作与射线结果，再重置到测试初态执行真实 MuJoCo 闭环。基准样本不驱动测试轨迹。

数值校验结果见 [parity.json](../logs/sim2sim/parity.json)：

| 对齐项 | 最大误差 |
|---|---:|
| Actor 输入（含历史、参考窗口） | 2.09e-7 |
| 同观测下策略动作 | 4.18e-7 rad |
| 刚体位置 FK | 4.55e-6 m |
| 高程射线命中位置 | 3.82e-6 m |
| raw 参考动作偏置 | 0 |

MuJoCo 的接触求解器、摩擦模型、隐式 PD 和关节速度约束处理与 PhysX 存在差异；本次没有声称两个引擎的动力学完全一致。尚未移植完整 MuJoCo 训练环境、奖励和终止管理器。

## 复现

在项目根目录执行。当前服务器已存在 contract；更换训练任务或控制参数时先重新导出：

```bash
bash scripts/pmt.sh mujoco-contract --output logs/sim2sim/contract.pt
```

独立 MuJoCo 评估（不需要 GPU 或 Isaac 运行进程；输出目录须为新目录）：

```bash
bash scripts/pmt.sh mujoco-eval \
  --contract logs/sim2sim/contract.pt \
  --motion-index 0 --steps 1499 \
  --output logs/sim2sim/my_mujoco_run
```

查看已完成的独立物理运行，可重用之前的 VNC、高程网格和热图面板：

```bash
bash scripts/pmt.sh replay logs/sim2sim/student_motion0 --loops 3

# 为新运行导出视频
bash scripts/pmt.sh replay logs/sim2sim/my_mujoco_run --video \
  --output logs/sim2sim/my_mujoco_video
```

`mujoco-eval` 实际运行策略和 MuJoCo 物理；`replay` 仅显示该次运行记录。评估程序正常退出表示评估完成，具体跟踪是否通过以 `metrics.json` 的 `tracking_passed` 为准。
