# 同动作跨地形足部受力对比

入口 `scripts/compare_pgmt_foot_forces.sh` 使用现有 Stage2 回放器，串行采集平地、楼梯、斜坡，再离线生成图表。用于判断接触力与落脚偏差的关系，当前不修改 reference、奖励或策略。

## 运行

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
bash scripts/compare_pgmt_foot_forces.sh \
  --checkpoint logs/pgmt_stage2_center2x2/train_03/model_17500.pt \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz \
  --device cuda:0 --level 1 --patch_id 5 \
  --direction up --spawn_offset_x 0 \
  --episodes 1 --max_steps 600 \
  --log_dir "logs/pgmt_force_comparison/$(date +%Y%m%d_%H%M%S)"
```

`--log_dir` 必须是新目录。每个地形最多运行 `--max_steps` 个控制步，且在完成 `--episodes` 个 episode 后提前停止；多个 episode 时应相应增大总步数。默认加载上述 Stage2 四头权重，Stage1 权重不能用于此入口。下楼／下坡使用 `--direction down --spawn_offset_x -2.35`，平地也采用相同局部出生偏移。

同一次对比固定 checkpoint、动作文件、seed、patch 编号、局部出生 XY 和朝向；关闭观测噪声、随机动力学、推扰和课程。地形导致的初始放置 Z 和后续物理轨迹可以不同。“相同动作”指相同 reference 和策略，不是强制机器人逐帧执行相同关节轨迹。出生点后仍须检查机器人是否真正进入台阶／坡面，短片段或提前失败可能只覆盖起始平台。

三个日志文件是 `flat.log / stairs.log / slope.log`；每个地形目录保留原有 `env.yaml / training_status.json / episodes.jsonl / trajectory.jsonl`，新增 `foot_forces.npz`。分析结果在 `comparison/`：

- `reference_aligned.png`：第一回合按 reference 时间和物理子步对齐的左右脚地形法向力世界 Z 分量、XY 模长，以及踝关节 link 原点的 Z 跟踪误差。
- `touchdown_aligned.png`：各地形都能匹配到、且窗口完整的同一次参考落脚，按各自真实触地时刻对齐；最多显示 8 次。没有共同完整事件时不生成此图，JSON 会明确报告数量为 0。
- `metrics.csv`：所有地形共同覆盖的 episode/frame/substep 上的均值、Z／XY 峰值、向上法向力冲量、实际支撑与参考标签不一致比例、相对平地的力向量 RMS 差异和踝位置 RMSE。默认排除每回合前 0.1 s；缺测不会填 0，积分缺测时结果留空。
- `touchdowns.csv / comparison.json`：触地时序、参考事件匹配、落脚误差、窗口是否完整和未匹配事件。未匹配参考事件可能来自漏踩，也可能来自时差超过匹配容差。`coverage` 保留每个地形的原始覆盖范围，避免把提前终止的曲线当成完整动作。

离线重画，无需启动 Isaac：

```bash
MPLCONFIGDIR=/tmp/pgmt-force-matplotlib \
/home/lzp/miniconda3/envs/pmt/bin/python scripts/pgmt_force_analysis.py \
  --run flat=logs/pgmt_force_comparison/RUN/flat \
  --run stairs=logs/pgmt_force_comparison/RUN/stairs \
  --run slope=logs/pgmt_force_comparison/RUN/slope \
  --output logs/pgmt_force_comparison/RUN/comparison
```

## 测量与时间约定

物理步长为 0.005 s，控制步长为 0.02 s。采集器在每次 `scene.update` 后读取力和足部运动学，共 200 Hz；不依赖奖励的 50 Hz 末子步采样。每步校验最后一个物理子步的力与 `contact_state(for_reward=True)` 一致；自动 reset 前采集的数据保留，reference 在控制步开始时冻结。暂停和渲染不添加样本。

NPZ 保存左右脚的全对象与地形过滤 XYZ 力、实际踝位置／速度／姿态、当前控制区间持有的 reference 足部位置／速度／姿态、参考支撑标签与有效性、实际／参考足下地形高度，以及 episode/frame/substep/time/done。`done` 表示所属控制步结束时 episode 终止，不能把它当成每个子步都发生了一次终止。

本机 Isaac Lab 的两个力通道都是**法向接触力合力，不含摩擦力**。斜坡上世界 XY 分量也可能来自坡面法向，不能直接叫作摩擦力或碰撞异常；世界 Z 也不等于坡面法向。冲量名称为 `upward_normal_impulse_ns`，积分的是 `max(Fz, 0)`，不是完整地面反力冲量。

分析用触地事件采用原有 10 N 向上地形力阈值，要求此前连续未支撑至少 0.04 s、随后支撑至少 0.01 s；时间戳取首次越阈值的物理采样端点。初始承重、短暂尖峰和未知测量不会伪造触地。此判据仅用于离线分析，不改变训练的接触状态机。

参考落地时间取 `swing → stance` 所在控制区间的起点。实际事件与参考事件在同回合、同脚内做保持顺序的一对一匹配，最大偏差默认 0.3 s；超限保留为未匹配。先匹配到参考事件，再比较不同地形，避免某次漏踩使后续事件整体错位。事件受力窗口为触地前 0.10 s 至触地后 0.15 s；若录制提前结束或有缺测，不输出该窗口的峰值和冲量。采样端点带来至多一个物理步量级的触地时间分辨率，reference 标签本身为 50 Hz。

若时间容差内存在多个候选实际触地或参考落脚，标为 `ambiguous_match`，图标题同时显示 `AMBIGUOUS time match`。这类匹配只用于诊断，不应直接作为“同一步”的修正监督。`event_summary` 列出未匹配实际／参考事件和歧义数量；`common_unambiguous_complete_touchdowns` 单独统计无歧义的共同完整事件。

落脚误差是实际触地时踝 link 原点与**匹配的参考触地帧**踝原点之差；不是鞋底接触点误差。曲线中的连续误差则使用当前持有的 reference。地形高度作为后续判断是否踏上台阶／坡面的证据保存，目前不从力差直接反推三维落脚修正量。

## 三栏对比视频

采集时加 `--record_video_states`，同时保存完整关节位置、刚体位置和姿态。旧的仅足部记录不能恢复完整机器人，需重新采集：

```bash
bash scripts/compare_pgmt_foot_forces.sh --record_video_states \
  --device cuda:0 --level 1 --episodes 1 --max_steps 600 \
  --log_dir logs/pgmt_force_comparison/NEW_VIDEO_RUN

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
/home/lzp/miniconda3/envs/pmt/bin/python scripts/render_pgmt_force_comparison.py \
  --source logs/pgmt_force_comparison/NEW_VIDEO_RUN \
  --output logs/pgmt_force_comparison/NEW_VIDEO_RUN/video --speed 0.5
```

视频为 1920×960 三栏，默认 0.5 倍速，三种地形按 reference 帧同步。左右脚分别标为蓝色和橙色，底部绘制全部 200 Hz 受力采样，数值显示当前 20 ms 的 Fz 峰值。必须是每种地形各一回合、参考帧序列完全相同的采集；提前终止造成长度不一致时明确报错，不拼接不同回合或补假帧。

渲染器只回放记录的 Isaac / PhysX 姿态，不重新运行策略或 MuJoCo 物理；地形使用同一 manifest 和 PLY 三角形，并校验哈希。每帧按名称映射关节并做独立 FK 检查，身体位置与记录差异超过 1 mm 即失败。`video_validation.json` 记录实际最大误差、编码帧数、时长和视频哈希。可先加 `--preview_only` 检查起始、中间和终止帧。

本次视频输出位于 [`20261103_video_clip1/video`](../logs/pgmt_force_comparison/20261103_video_clip1/video)，同目录保留三张预览图与校验结果。

## 验证

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_force_comparison.py \
  tests/test_pgmt_contact_state.py tests/test_pgmt_contact_force.py
```

覆盖物理子步冲量单位、漏踩的一对一匹配、reset 时序、初始接触／尖峰／未知样本、截断窗口、跨回合隔离、不同动作／权重／采样配置拒绝配对和图表导出。真实回放还必须检查 `.log` 中没有 PhysX 接触缓冲溢出或 CUDA 错误。采集模式采用单机器人缓冲容量并写入元数据，不更改求解器参数；普通回放和训练不受影响。

## 2026-11-03 首次对比

正式结果：[`logs/pgmt_force_comparison/20261103_train03_clip1_l1`](../logs/pgmt_force_comparison/20261103_train03_clip1_l1)。使用本页命令中的 `train_03/model_17500.pt`、`walk1_subject1_clip1.npz`、seed 1017、L1/patch 5。实际台阶高 0.09894 m，坡度约 5.87°。三个地形各完整完成 499 控制步（9.98 s）、1996 个物理子步，力与位姿全部有限；每控制步末尾力与冻结奖励快照一致。楼梯左右脚足下地面均上升约 0.495 m；斜坡左／右脚分别上升约 0.240 m／0.219 m。

以下为排除初始 0.1 s 后、共同 9.88 s 窗口内的世界 Z 地形法向力峰值，不是同一次触地峰值的配对统计：

| 地形 | 左脚峰值 N | 右脚峰值 N | 支撑标签不一致比例（左／右） |
| --- | ---: | ---: | ---: |
| 平地 | 1437.8 | 1743.0 | 14.1% / 13.1% |
| 楼梯 | 1541.5 | 2366.7 | 17.2% / 15.5% |
| 斜坡 | 919.9 | 1529.5 | 15.1% / 14.5% |

三者共有 8 个匹配且窗口完整的参考落脚，其中 6 个在三种地形上均无时间匹配歧义。楼梯有 1 个参考事件在 ±0.3 s 内未找到匹配，另有 2 个匹配存在多个候选；不能把这两类事件直接用于修正律拟合。详细 CSV、图表和 JSON 位于结果的 `comparison/` 目录。这是单动作、单 seed 的初步测量，不代表跨动作统计结论。

代码验证为 **57 passed、2 skipped**（测试进程无 CUDA；真实采集单独使用 GPU 0）。正式三个回放日志无 PhysX/CUDA 错误，存在宿主机 inotify watch 数量不足的启动提示，不影响本次采样与校验。旧默认 `train_01/clip0` 的采集验证保留在 `20261103_l1_small`，与正式结果分开。
