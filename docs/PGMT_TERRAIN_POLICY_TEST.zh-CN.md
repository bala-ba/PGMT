# Stage2 policy 多地形可视化测试

入口仍为 `scripts/play_pgmt_stage2_stairs.sh`，兼容之前的楼梯命令。增加
`--terrain flat|stairs|slope|boxes|rough`，默认 `stairs`。使用已训练的感知策略进行确定性推理。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
DISPLAY=:1 bash scripts/play_pgmt_stage2_stairs.sh \
  --checkpoint /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_stage2_center2x2/train_01/model_19999.pt \
  --motion /home/lzp/projects/perceptive_tracking/PMT/assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz \
  --terrain slope \
  --level 1 \
  --direction up \
  --patch_id 5 \
  --speed 1 \
  --max_episodes 0 \
  --log_dir "logs/pgmt_terrain_policy/$(date +%Y%m%d_%H%M%S)"
```

## 界面操作

- `FLAT / STAIRS / SLOPE / BOXES / ROUGH`：选择地形并重置到该区域。
- `L0`–`L9`：选择难度并重置，保留当前地形。
- `UP from bottom / DOWN from rim`：仅楼梯、斜坡启用。上行从坑底中心开始；下行从中心 X 方向偏移 −2.35 m 处开始。参考轨迹的净位移对齐世界 +X，原动作的转向、侧向位移和关节轨迹保留。
- `Alternate up/down`：仅楼梯、斜坡启用。开启时，原生回合结束后交替上下行；`--no_alternate` 可以关闭。
- 平地、箱体、粗糙地形从所选 6 × 6 m 小块中心开始；`--direction` 对它们不起作用。
- `--patch_id 0..15` 选择 4 × 4 区域中的小块，默认 5。编号由已有地形 manifest 定义。
- `Pause / Run`、`Single step`、速度、感知显示、参考机器人显示开关沿用原入口。暂停状态切换地形后仍保持暂停。

白色机器人执行 policy 和物理仿真；青色机器人只显示当前 `PGMTState.reference()`，默认沿世界 Y 偏移 1 m 作对照，不参与物理。`Reference overlay` 可取消显示偏移。参考轨迹保留重置时的统一变换，不逐帧贴地，可能穿过或离开地表。

## 切换和记录

切换地形/难度通过已有完整 `env.reset()`：先更新实际地形类型、等级和出生 origin，并按出生 XY 查询真实高度，再建立参考动作的统一变换。机器人、command、未来帧、接触状态和观测历史沿用原生 reset 链路。相机、绿色轨迹和青色参考随后更新到新位置。

手动切换不会被记为一个成功或失败回合。动作没有增加距离、直线程度或测试终点限制，仍使用完整参考动作和训练任务的原生回合终止条件。此入口关闭感知增强、观测噪声、随机动力学、推力和地形课程更新；不进行 PPO 更新。

显式 `--motion` 选择优先于训练用动作/地形兼容表：测试入口在本次日志目录生成 `playback_motion_compatibility.csv`，仅允许指定动作并对五种地形全部放行。动作从第 0 帧开始，不要求起始帧存在有效支撑标签。训练原兼容表和采样器不变；物理初始化与原生回合终止条件仍按上述规则执行。`training_status.json` 中 `motion_terrain_filter=false` 记录这一测试配置。

日志目录中的 `spawns.jsonl` 记录每次重置的地形、难度、小块、出生高度、实际/参考 root 和起始帧；`trajectory.jsonl` 记录逐步轨迹；`episodes.jsonl` 记录自然结束的回合；`training_status.json` 记录当前选择、加载结果和运行状态。截图文件名包含地形名。

## 2026-10-17 验收

使用 `model_19999.pt`、`walk1_subject1_clip1.npz`、CUDA 1 和 Isaac Sim 6.0.1 可视化入口，在同一进程中调用实际界面的选择回调，共运行 520 个控制步。覆盖 L1 的五种地形、stairs/slope 两个方向，以及 rough 从 L1 切换到 L9。每次选择先暂停、重置并检查，再恢复运行。

- 8 次出生均从第 0 帧开始，实际 root 与参考 root 对齐；真实位置查询的地形、难度、小块与选择一致。
- 21 × 21 扫描全部有效，扫描与网格高度查询最大误差 `3.8147e-6 m`；网络观测与当前扫描后处理结果一致。
- 暂停切换时，感知点、参考机器人和相机刷新到新地形；青色参考含 0 个物理 prim。
- 手动切换未计入成功/失败回合；policy 参数和源 checkpoint 未改变。
- 初始楼梯上行的前 100 步与修改前对照日志逐步比较，实际 root 和参考 root 的最大差异均为 0，参考帧和 done 标记完全一致。
- 首次运行后切换暴露原有手动 reset 的 `Inplace update to inference tensor outside InferenceMode` 错误。修复为手动 `env.reset()` 与 `env.step()` 使用相同的 `torch.inference_mode()`，最终验收正常退出。

结果：`logs/pgmt_terrain_policy_validation_20261017/switches_03/validation_summary.json`。
逐次检查：同目录 `switch_validation.json`、`spawns.jsonl`；截图：`validation_*.png`。
验收驱动：`logs/pgmt_terrain_policy_validation_20261017/validate_switches.py`，只在内存中插入界面选择回调，不改写产品脚本。

这是地形选择、重置与可视化功能验收，未重新评估全部地形/难度的完整回合成功率。GUI 使用本机现有 Isaac Sim 6.0.1 兼容入口；原训练环境为 Isaac Sim 5.1.0。

## 指定动作被训练兼容表拦截的修复（2026-10-17）

`dance1_subject1_clip1.npz` 在原训练兼容表中排除了 `stairs`。旧测试入口只加载该动作后仍读取训练表，因此楼梯可采样动作数为 0，抛出 `No compatible motion with a valid reset start`；并非该动作缺少第 0 帧或有效支撑导致。现按上述测试专用表放行指定动作，训练兼容表不变。

- 使用用户指定的 checkpoint、dance 动作、stairs L1、up、patch 5，在 DISPLAY=:1 / CUDA 1 中完成 120 个控制步，正常退出；起始帧为 0，实际和参考 root 完全对齐。
- 严格加载 checkpoint，结束时 policy 参数和源 checkpoint 均未改变。本次只验证启动与短时推理，不代表完整动作成功。
- 用本次生成的实际 CSV 调用共享加载/采样函数，五种地形均可选择唯一动作的第 0 帧；原训练表仍拒绝此动作的 stairs 组合。
- 原采样与小块布局测试共 25 项通过。
- 验证日志：`logs/pgmt_playback_compatibility_validation_20261017/dance_stairs_01/`，含 `training_status.json`、`compatibility_validation.json`、出生和逐步记录，以及开始/第 100 步截图。
