# PGMT stage 1：PMT 原生训练接入

## 入口和范围

Gym task：`PMT-PGMT-Stage1-G1-v0`；配置：[`pgmt_stage1.yaml`](../configs/task/pgmt_stage1.yaml)。

基准为用户确认的 **2B：`PMT-G1-MultiMotionV2-Flat-v0`**。本轮完成代码接入和短训练验证，未启动正式长训练。

沿用 PMT 的 Isaac Lab 环境、G1、`MultiMotionCommandV2`、`bin_adaptive`、eager FP32 数据加载、4000 帧 chunk、随机化、重置、终止条件、PPO、OnPolicyRunner 和 checkpoint 流程。动作仍为：

```text
q_target = q_default + G1_ACTION_SCALE * action
```

新增的是 PGMT 网络及输入、PGMT 奖励、三价值头所需的 PPO/storage 适配、LAFAN1 格式转换和原生 CPU MuJoCo 策略分支。

| 项目 | 接入方式 |
| --- | --- |
| 网络 | 历史 cross-attention + RoPE intent attention；embedding 128、4 heads；MLP 512/256/128 |
| 观测 | 当前 96、历史 10×96、未来命令 8×61、critic 251 |
| 未来帧 | 0/1/3/7/15/31/63/127，限制在当前原生 chunk 内 |
| critic | `upper / lower / auxiliary` 三头，独立 GAE，优势求和后按 PMT 原生方法归一化 |
| 奖励 | 本地 PGMT optimized_v2 的 21 项权重和公式，由 Isaac RewardManager 积分一次 dt |
| reset | 原生 PMT；不启用倒地池及依赖该模式的 recovery upward velocity 奖励 |
| 归一化与探索 | PMT 原生模块、更新时机和 Gaussian 配置 |
| 数据 | 全部 40 条 LAFAN1 G1 重定向动作，441,140 帧，50 Hz |

环境继续返回标量 reward；`extras["reward_groups"]` 提供 `[N,3]` 给 PPO。三组贡献通过原生 logger 的 `Reward_Groups/*_per_step` 记录；各项仍为 `Episode_Reward/*`。

观测末 29 维读取 `robot.data.joint_pos_target`。原生 command reset 不重设该 target，适配层遵守这一行为。command 内部换片会清理 PGMT 历史和奖励缓存，保持原生 episode mask。

原 command 仍使用 14 个 tracked bodies；PGMT 全身奖励另建一个**原生 MotionDataStore**，加载相同文件并严格核对 chunk。实测 136 个 chunk，额外 tensor 内存约 **0.908 GiB**。

## 准备数据

在 PMT 目录、已安装的 `pmt` 环境中运行：

```bash
source scripts/pmt_env.sh
python scripts/prepare_pgmt_lafan.py
```

默认源为 `../PGMT/assets/motions_50hz`，输出 `assets/motions/pgmt_lafan1_50hz`。转换仅重排名称和修改格式；不重采样、不改写源文件。输出已存在时会拒绝覆盖；有意重建时使用 `--overwrite`。

`manifest.json` 记录文件哈希、转换程序/模型哈希、帧数、名称映射和 FK 校验。数据目录被 git 忽略。可以用 `PMT_PGMT_MOTION_ROOT` 指定转换后的目录，也可通过原生训练参数 `--motion_file` 覆盖。

## 训练和恢复

本轮使用的短验证规模：

```bash
bash scripts/pmt.sh train \
  --task PMT-PGMT-Stage1-G1-v0 \
  --num_envs 8192 --max_iterations 20000 \
  --headless --device cuda:0 \
  --log_dir logs/pgmt_stage1/pgmt_8192x20000
```

训练参数保留原生 rollout 24、5 epochs、4 minibatches、初始 LR 1e-3、初始动作标准差 1.0。解析后的完整配置保存在运行目录的 `params/`。

20 次更新的原生 checkpoint 名为 `model_19.pt`。恢复示例只执行一次额外更新：

```bash
bash scripts/pmt.sh train \
  --task PMT-PGMT-Stage1-G1-v0 \
  --num_envs 8192 --max_iterations 20000 \
  --resume_path /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_stage1/pgmt_8192x20000/model_12200.pt \
  --headless --device cuda:0 \
  --log_dir /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_stage1/pgmt_8192x20000
```

checkpoint 使用 PMT 的 `model_state_dict / optimizer_state_dict / iter / policy_metadata`。保留原生恢复语义：不恢复仿真现场，也不承诺保存 2B bin sampler 统计。原生 runner 从当前 iteration 标签开始，所以恢复一次更新后标签可能仍为 19；本轮通过 Adam step **400 → 420** 和 49 个模型 tensor 改变确认更新实际发生。

## Isaac 接口检查

在 PMT 的 Python 环境中：

```bash
source scripts/pmt_env.sh
python scripts/validate_pgmt_stage1.py \
  --headless --device cuda:0 --num_envs 4 --steps 64 \
  --checkpoint logs/pgmt_stage1_validation/pgmt_64x20/model_19.pt \
  --output logs/pgmt_stage1_validation/interfaces.json \
  --resume-update
```

检查真实环境中的奖励分组、动作目标、历史推进、重复读取、command 内部换片、future clamp、sampler outcome 和原生 checkpoint 恢复。数值检查关闭观测噪声，保留原生物理、随机化和重置；正式训练配置不受影响。

## CPU MuJoCo

复用 PMT 已有的 model builder、CPU `mj_step`、评测和可视化入口。首先从真实 Isaac 环境导出控制参数及数值样本：

```bash
bash scripts/pmt.sh mujoco-contract \
  --task pgmt_stage1 \
  --checkpoint logs/pgmt_stage1_validation/pgmt_64x20/model_19.pt \
  --output logs/pgmt_stage1_validation/pgmt_contract.pt \
  --steps 3 --device cuda:0

bash scripts/pmt.sh mujoco-eval \
  --contract logs/pgmt_stage1_validation/pgmt_contract.pt \
  --motion-index 0 --steps 100 \
  --output logs/pgmt_stage1_validation/mujoco_example

bash scripts/pmt.sh mujoco-live \
  --contract logs/pgmt_stage1_validation/pgmt_contract.pt \
  --motion-index 0 --steps 100 \
  --output logs/pgmt_stage1_validation/mujoco_live_example
```

`--motion-index` 按 contract 中 40 条源文件顺序索引，不使用内部 136 个 chunk 的编号。也支持 `--motion-file`。未来窗口遵循原生 chunk 边界；离线 eval 到边界停止，live 到边界重新开始所选片段。

export 关闭观测噪声和随机化事件，用于验证同状态观测/动作计算；训练设置不变。CPU 初始历史使用导出的 Isaac 实际初始 target。独立评测不导入 Isaac，不需要 GPU；它与训练的物理引擎不同，同状态数值通过不代表两引擎动力学完全相同。

## 本轮验证结果

报告目录：[`logs/pgmt_stage1_validation`](../logs/pgmt_stage1_validation)。

| 检查 | 结果 |
| --- | --- |
| 数据转换 | 40 条、441,140 帧；872 个抽查状态含 chunk 边界；FK 最大位置误差 8.17e-7 m |
| CPU 数值及兼容测试 | 341 项通过，包含 PGMT 原函数比较、三头 GAE/PPO、单头回归及 MuJoCo 接口 |
| 原生任务回归 | 原 2B 任务加载同批数据，4 环境、1 次 PPO 更新成功 |
| PGMT 短训练 | 64 环境 × 20 更新 × 24 rollout = 30,720 步；全部日志数值及 checkpoint tensor 有限 |
| 三组奖励与总奖励 | 真实 Isaac 最大差 2.98e-8 |
| 动作目标 / 最新历史帧 | 真实 Isaac 最大差均为 0 |
| 重置 / 采样 | 环境 done、command 内部换片、history 清理、future clamp 和 sampler outcome 检查通过 |
| checkpoint 恢复 | 原生 PPO 额外一次更新成功，Adam step 400 → 420 |
| Isaac/MuJoCo 同状态 | 3 个真实样本；观测最大差 7.45e-7、动作 1.04e-7、目标角 4.58e-8 rad |
| MuJoCo 全数据短闭环 | 全部 40 条各 100 控制步；40/40 运行及接口检查通过，无数值重置或 solver warning |
| MuJoCo 原生窗口 | 临时 Xvfb + Mesa，10 控制步、40 次物理步进、5 次窗口更新，正常退出 |

**短训模型尚无稳定跟踪能力**：上述 40 条短闭环均未通过现有 tracking diagnostic。该模型用于验证数据、网络、优化和仿真接线，不能用本轮结果评价正式训练性能。LAFAN1 本身含地面动作，绝对高度跌倒诊断也可能触发；评测同时保留相对参考高度损失。

关键证据：`data_summary.json`、`training_summary.json`、`interfaces.json`、`resume_summary.json`、`mujoco_all40/summary.json`。源码指纹保存在 `source_fingerprints.json`；初始基线保存在 `baseline.json`。

## 文件导航

- 网络：[`pgmt_actor_critic.py`](../motion_tracking_rl/networks/pgmt_actor_critic.py)
- 环境：[`stage1.py`](../pmt_tasks/env_cfgs/pgmt/stage1.py)、[`pgmt_runtime.py`](../pmt_tasks/pgmt_runtime.py)
- 共用观测/奖励数学：[`pgmt_math.py`](../pmt_tasks/pgmt_math.py)
- 三头支持：[`ppo.py`](../motion_tracking_rl/algorithms/ppo.py)、[`rollout_storage.py`](../motion_tracking_rl/storage/rollout_storage.py)
- 原生 MuJoCo 适配：[`mujoco_standalone.py`](../pmt_tasks/backends/mujoco_standalone.py)
- 已确认的设计选择：[`PGMT_STAGE1_PLAN.zh-CN.md`](PGMT_STAGE1_PLAN.zh-CN.md)
