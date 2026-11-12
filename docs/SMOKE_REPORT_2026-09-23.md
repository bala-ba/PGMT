# PMT 服务器验收报告 · 2026-09-23

已完成独立 conda 环境重建、运行问题修复、训练与测试 smoke test，以及基于 `projects/tutorial` 的 VNC 可视化和 MP4 录像。

**最终验收通过**：四项训练、五项回放均完成；入口进程退出码为 0，严格结果检查 `smoke_suite_passed=true`。正式结果位于 [验收汇总 JSON](../logs/smoke/2026-09-23-acceptance/summary.json)，完整执行记录见 [验收日志](../logs/setup/smoke-acceptance.log)。其他日期后缀目录为调试记录，不作为最终验收结果。

## 新环境

| 项目 | 已验证配置 |
| --- | --- |
| Conda | `pmt`，路径 `/home/lzp/miniconda3/envs/pmt` |
| Python | 3.11.16 |
| Isaac Lab / Isaac Sim | 2.3.2 官方 pip 包 / 5.1.0.0 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / CUDA 12.8 |
| TensorDict / NumPy | 0.9.0 / 1.26.0 |
| 可视化 | MuJoCo 3.3.5 + Mesa + tutorial Xvfb/VNC |
| GPU | 两张 RTX 5090，均通过 CUDA 矩阵运算；训练 smoke 使用 GPU 0 |

`pip check` 无依赖冲突；单元测试 **543 passed, 16 skipped**。跳过项涉及可选后端、外部仓库依赖，以及需要在 SimulationApp 启动后导入的 Isaac Lab 配置；核心仿真另外执行了下述真实运行检查。

可重建配置：[environment.yml](../environment.yml)、[requirements-pmt.txt](../requirements-pmt.txt)。完整快照：[pip freeze](../logs/setup/pmt-pip-freeze.txt)、[conda export](../logs/setup/pmt-conda-export.yml)、[版本与 GPU 检查](../logs/setup/versions.json)。原有 conda 环境未改动。

## 训练与测试

使用仓库随附的 99 对 `walk_dance1sub2start` raw/optimized 示例动作和 big_map 地形。平地任务使用 raw 动作；感知任务通过 `--motion_file` 指定随附 optimized 数据。

每项训练均检查 TensorBoard 标量、checkpoint 张量为有限值，并确认首尾 checkpoint 之间有权重更新。

| 训练路径 | 迭代数 | 发生更新的 weight 张量数 | 最终 checkpoint |
| --- | ---: | ---: | --- |
| 平地 PPO | 5 | 8 | [model_4.pt](../logs/smoke/2026-09-23-acceptance/flat_train/model_4.pt) |
| 感知 motion-token PPO | 2 | 18 | [model_1.pt](../logs/smoke/2026-09-23-acceptance/perceptive_train/model_1.pt) |
| 真实教师 → 视觉学生蒸馏 | 2 | 43 | [model_1.pt](../logs/smoke/2026-09-23-acceptance/distill_train/model_1.pt) |
| 新蒸馏学生 → PPO 微调 | 2 | 50 | [model_1.pt](../logs/smoke/2026-09-23-acceptance/finetune_train/model_1.pt) |

平地训练使用 8 个并行环境，其余三项使用 4 个；种子为 42。蒸馏加载随附 `walkdance_bigmap_teacher.pt`，微调加载本次新生成的蒸馏 checkpoint。

五项回放分别加载新训练模型与已有预训练模型，检查动作、观测、奖励均无 NaN/Inf。重置数合计所有并行环境；body error 是该次短回放中的 `error_body_pos` 均值。

| 回放 | 环境数 | 控制步数 | 重置数 | body error (m) |
| --- | ---: | ---: | ---: | ---: |
| distill_trained | 2 | 100 | 10 | 0.0762 |
| finetune_trained | 2 | 100 | 7 | 0.0671 |
| flat_pretrained | 2 | 300 | 0 | 0.1211 |
| flat_trained | 2 | 200 | 17 | 0.1688 |
| perceptive_trained | 2 | 200 | 16 | 0.0961 |

这些是短时可运行性检查，不代表模型收敛。新模型仍会失败重置；这里没有宣称达到论文指标或原始全量数据的训练效果。

训练曲线：[平地 PPO](../logs/smoke/2026-09-23-acceptance/flat_train/training_curves.png) · [感知 PPO](../logs/smoke/2026-09-23-acceptance/perceptive_train/training_curves.png) · [蒸馏](../logs/smoke/2026-09-23-acceptance/distill_train/training_curves.png) · [PPO 微调](../logs/smoke/2026-09-23-acceptance/finetune_train/training_curves.png)。各训练目录保留原始 TensorBoard events 和最终 env/agent YAML。

## 可视化结果

复用 `~/projects/tutorial/isaacsim_gui.sh display --no-hints` 和 `MUJOCO_GUI.md` 的 Mesa 显示方式。VNC 地址为 `100.122.53.37:5900`，密码沿用 tutorial 配置。训练与测试的 VNC 窗口均已实际显示并截图。

当前服务器上的 Isaac Sim 5.1 原生 RTX 窗口/离屏渲染存在 Vulkan 呈现错误及 `librtx.scenedb.plugin.so` 崩溃。因此使用 **Isaac Lab 计算物理、训练和推理，独立 MuJoCo 进程显示真实状态** 的方案。显示进程仅执行正运动学，不执行 MuJoCo 物理步进，也不替换策略动作。地形来自与 Isaac Lab 相同的 STL。

30 个机器人刚体逐帧核对，全部录像中的最大位置差小于 0.007 毫米。视频均为 **960×640、50 FPS**，已核对视频帧数与实际记录状态数一致，且画面非空、机器人确实运动。验收数据见 [录像检查 JSON](../logs/visualization/verification.json)。

| 内容 | 时长 | 文件 |
| --- | ---: | --- |
| 平地 PPO 真实训练，额外 12 轮、4 环境 | 5.76 秒 | [training.mp4](../logs/visualization/flat_train_verified/videos/training.mp4) |
| 预训练地形教师测试，300 步、0 次重置 | 6 秒 | [testing.mp4](../logs/visualization/terrain_teacher/testing.mp4) |
| 同一教师测试轨迹，较高视角重新渲染 | 6 秒 | [testing.mp4](../logs/visualization/terrain_teacher_replay/testing.mp4) |
| 本次新蒸馏/微调学生测试，150 步、7 次重置 | 3 秒 | [testing.mp4](../logs/visualization/new_finetuned_student/testing.mp4) |

学生录像使用修复验证时生成的 `logs/smoke/finetune-path-fix/model_1.pt`，与最终验收目录中的重新训练模型分开保存。教师短回放 body error 均值约 0.0253 m，不能据此代替完整评估。

截图：[训练 VNC 桌面](../logs/visualization/training_vnc_desktop.png) · [测试轨迹 VNC 回放桌面](../logs/visualization/testing_vnc_desktop.png)。每段录像旁还保存 `metadata.json`、`states.jsonl`、`render_result.json` 和预览图。

## 修复内容

- 新建适配 RTX 5090 的 Python 3.11 / CUDA 12.8 环境，锁定 Isaac Lab、Isaac Sim、NumPy、TensorDict 等依赖并消除版本冲突。
- 下载并验证所需 G1 机器人资源；通过项目启动脚本统一动作、地形、模型资源路径。
- 修正 editable 安装遗漏 `pmt_tasks`；训练与测试入口支持通用任务覆盖，动作路径在构建配置前应用，环境和策略使用一致设备。
- 修正 PPO 微调配置中不存在的 `mocap_house_rollout` 中间目录，与蒸馏共享 paired-motion 配置，并加入回归测试。
- 修正 Isaac Sim 快速关闭吞掉异常退出状态的问题。故意指定缺失动作文件时，已验证显示 `FileNotFoundError` 并返回退出码 1，见 [负向检查](../logs/setup/expected-failure.log)。严格汇总检查也能拒绝缺少微调阶段的旧调试运行。
- 增加配置保存、训练曲线、回放指标、VNC 实时显示、视频/状态保存及无需 GPU 的轨迹回放。

## 直接使用

在 PMT 仓库根目录执行，启动脚本会自动激活 `pmt` 并配置路径：

```bash
# 完整复跑 smoke test（每次使用新输出目录）
bash scripts/pmt.sh smoke

# 用已保存的真实测试状态进行 VNC 回放，无需 GPU 仿真
bash scripts/pmt.sh replay logs/visualization/terrain_teacher --loops 3

# 重新运行预训练教师，在 tutorial VNC 中实时显示并录制
bash scripts/pmt.sh gui-play --task PMT-WalkDanceBigMap-G1-v0   --resume_path checkpoints/pretrained/walkdance_bigmap_teacher.pt   --num_envs 1 --max_steps 500 --video --video_length 500   --video_dir logs/visualization/my_teacher_test
```

完整环境重建、训练、可视化及 TensorBoard 命令见 [SERVER_SETUP.md](SERVER_SETUP.md)。本次完成环境及核心流程的 smoke 验收；未执行长时间收敛训练、全部约 25 个任务逐一验证或完整论文复现。原生 Kit RTX 渲染仍是已知限制，当前提供的 VNC/录像路径已验证可用。
