# Stage 1 平地行走可视化

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
DISPLAY=:1 bash scripts/play_pgmt_stage1_flat.sh \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz \
  --speed 1 \
  --log_dir "logs/pgmt_stage1_flat_policy/$(date +%Y%m%d_%H%M%S)"
```

使用原生 `pgmt_stage1` 环境、`MultiMotionCommandV2` 和无感知三头策略；按 checkpoint 元数据重建网络并严格加载全部参数与归一化统计。没有迁移到 Stage 2，也没有增加地形感知输入。actor 使用确定性均值动作，不运行 PPO。

- 白色机器人：Stage 1 policy + 真实物理仿真。
- 青色机器人：当前 `PGMTState.reference()` 的纯视觉对照，无碰撞、刚体或关节；默认只在显示上沿世界 Y 平移 1 m。
- `Reference overlay`：取消对照的显示偏移；`Reference on / off`：开关参考。
- `Pause / Run`、`Single step`、`Reset to frame 0`、速度按钮、跟随/自由相机均在独立窗口中。
- `--max_steps N`：有限步检查后退出；默认 0，持续显示。
- `--hide_reference`：初始隐藏参考；`--reference_overlay`：初始使用相同世界坐标显示参考。

动作固定从第 0 帧开始，关闭初始位姿/速度/关节噪声、观测噪声、随机动力学和推力。保持 Stage 1 原有动作控制、参考坐标系和终止条件，包括 15 秒回合上限。原生 Stage 1 在动作耗尽时由 command 内部重新采样并返回第 0 帧，不自动生成 done；本入口保留这一回放行为。参考动作不施加 Stage 2 的地形摆放或轨迹转向。

本机 GUI 使用 `isaacsim` 环境的 Isaac Sim 6.0.1、物理 CUDA 1、`DISPLAY=:1`，沿用已有的 Isaac Lab 兼容层。Stage 1 原训练环境为 Isaac Sim 5.1.0，因此这次可视化不能视作跨版本完全相同的动力学复现。

## 本机平地资源处理

默认平面加载器依赖 `Isaac/Environments/Grid/default_environment.usd`，当前机器解析到无效的 `None/Isaac/...` 路径。仅在此回放入口中使用 `PhysicsSchemaTools.addGroundPlane`（Isaac 原生 GroundPlane 使用的同一接口）创建无限平面碰撞体及显示网格，再绑定原生 Stage 1 的地面物理材质：静/动摩擦系数 1、恢复系数 0、组合方式 multiply。原始训练配置和通用地形加载器不改动。

## 记录

`training_status.json` 记录加载检查、观测形状、帧号、位置误差和进程状态；`trajectory.jsonl` 记录逐步实际/参考 root；`endings.jsonl` 区分动作内部耗尽与原生回合终止；截图自动保存在日志目录。动作内部循环仅记为 `reference_finished`，不作为课程成功或成功率统计。

2026-10-17 当前可视化日志：`logs/pgmt_stage1_flat_policy_20261017/viewer_04`。确认严格加载 `model_15000.pt`，`terrain_mode=disabled`，三头为 upper/lower/auxiliary；观测为 proprio 96、history 10×96、command 8×61、critic 251。源 checkpoint 哈希检查未改变。

首轮物理回放（`viewer_02`）完整运行 499 步参考动作后正常内部循环，anchor_pos、anchor_ori、ee_body_pos 均未触发。随后正常触发原生 750 控制步的 15 秒回合超时。

显示验收发现超大显示网格与可见碰撞 Plane 存在条纹，改为 100 m 显示网格、隐藏碰撞 Plane 的渲染，并增加 1 m 间距的纯视觉网格线；无限平面碰撞仍启用。`viewer_04/step_100.png` 已检查显示正常。显示修复前后前 100 步的 root 与参考帧做逐步对比，结果记录在 `logs/pgmt_stage1_flat_policy_20261017/validation_summary.json`。

本机兼容问题也已记录：旧版材质创建包装器在 Sim 6 中签名不兼容，显示层改用原生平面颜色；旧测试实例退出时 Kit 的信号处理导致挂起，核实身份后清理。当前入口把退出信号处理放在 AppLauncher 之后安装，避免被 Kit 启动覆盖。旧测试实例已经退出，当前保留 `viewer_04` 供人工检查。
