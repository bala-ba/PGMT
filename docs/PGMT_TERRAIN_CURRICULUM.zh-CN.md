# PGMT 地形课程与续训恢复

2026-10-10：已接入 `PGMTPlacementEnvCfg` 的原生 `CurriculumManager`，激活项为 `terrain_progress`。实现位于 [pgmt_curriculum.py](../pmt_tasks/pgmt_curriculum.py)。本项完成课程、reset 与保存恢复接口；感知、地形奖励和完整 stage 2 训练配置继续按总计划实施。

## 独立成功判据

课程读取实际 termination 原因和本回合参考进度，不读取 AdaptiveSampler 的 success 计数。每个环境按**出生分配的地形类型和等级**结算；跨 tile 只影响当前位置查询，不能改变课程归属。

| 回合结果 | 独立判据 | 等级变化 |
| --- | --- | --- |
| `success` | 从第 0 帧开始，同一动作到达最后一帧，实际控制步数不少于 `max(1, motion_length-1)`，且没有跟踪失败、地图越界或其他截断原因 | +1 |
| `failure` | 当前跟踪失败终止项触发，即 `termination_manager.terminated=True` | −1 |
| `map_exit` | 地图越界 truncation，没有同时发生跟踪失败 | 不变 |
| `timeout` | 普通时间上限先于动作完成，没有失败或越界 | 不变 |
| `incomplete_motion` | 虽然触发末帧标志，但从中间帧开始、动作 ID 改变或实际运行时长不足 | 不变 |
| `other_truncation` | 其他已配置的 truncation 原因 | 不变 |

优先级为 **失败 → 地图越界 → 其他截断 → 动作完成/不完整 → 普通超时**。末帧同时失败只降级一次；完整动作与普通时间上限恰好同时发生可算完成。等级在 **L0–L9** 封顶，不采用 Isaac 原生“完成最高级后随机退回其他级”的策略。

2026-10-15：Stage2 的独立 `time_out` 终止项增加 2 控制步边界容差；当前第 498/499 步且动作已到末帧时也置真，未到末帧仍须满 500 步才超时。课程仍按上述优先级只记一次 success，所以 `TerminationCount/time_out` 与课程 `timeout` 可以不同；具体公式和验收见 [终止说明第 5 节](PGMT_STAGE2_TERMINATIONS.zh-CN.md#5-2026-10-15保留-10-s-上限增加控制周期边界容差)。

“实际控制步数”取 `common_step_counter - episode_start_step`，与 runner 可以随机化的 `episode_length_buf` 独立。采用 `length-1` 下限，是因为 Isaac 在自动 reset 后会立即推进一次新 command：机器人从第 0 帧重置，返回的 command 已是第 1 帧。仅手工将索引跳到末帧不能获得升级。

2026-10-14 已按用户指定更新 Stage2 失败条件：pelvis XY 轨迹误差 >1 m、相对当前地形的 torso 高度误差 >0.4 m、双手腕相对 pelvis 高度误差 >0.4 m，以及原姿态判据；超时为 10 s/500 步。详见 [终止说明](PGMT_STAGE2_TERMINATIONS.zh-CN.md)。奖励容差已实现，读取实际 torso 当前 tile 的 family/level；课程升降级和统计仍按出生分配。前四项跟踪失败连续超限 0.2 s（10 控制步）才终止，各项独立计时，恢复正常或 reset 清零；不随等级变化，没有额外 reset 宽限。课程仅在实际终止/截断时结算，不把容忍期内的原始超限计为失败。

## reset 顺序与防重复结算

```text
物理步 → termination → 旧回合 reward
  → 课程判据与统计（按旧出生 family/level）
  → 更新等级和 env_origins
  → sampler 记录旧 start_motion_id 的 outcome
  → 按出生地形兼容表采样新动作
  → 按新 origin 放置并重置机器人
  → 登记新回合起点 → command 推进 → 新观测
```

课程更新只改等级，地形类型列保持出生分配。`env_origins` 更新为 `terrain_origins[new_level, assigned_family]`；v5 在该原点附近放置，v6 在新出生区域的允许小块中随机选择 XY/yaw，再按真实地形和全身碰撞确定 Z，不直接把区域原点作为机器人位置。2026-10-15 起，Stage2 使用 `placement_patch_grid=(2,2)`，仅允许中央四块 5/6/9/10，历史 P1 默认仍为全 16 块；该设置进入课程 checkpoint identity。详见 [参考初始化说明](PGMT_REFERENCE_PLACEMENT.zh-CN.md)。动作级自适应、CSV 禁配过滤、`fixed_start_frame=0` 和关闭足部支撑过滤继续生效。

2026-10-11 已实际加载 v6 `model_15999.pt`，验证课程/采样/优化器的精确恢复、恢复后 PPO 更新及二次保存加载，见 [v6 P1 补充验收](PGMT_V6_P1_ACCEPTANCE.zh-CN.md)。这会从第 0 帧重新初始化仿真 episode，不逐位续接中断的物理轨迹或 RNG 序列。

每个回合有 active、起始步、起始动作/帧和待结算标记，支持局部 reset。初始化、手动中断、同一步重复 reset 和 checkpoint 恢复不会产生虚假的成功/失败，sampler 也不重复计数。自然结束后仍保留原生动作 sampler 的 `~terminated` success 语义，因此地图越界/超时可能被 sampler 记为 success；**地形升级始终使用上表的独立判据**。

动作耗尽和地图越界仍通过 `time_out=True` 返回 truncation，wrapper 的 done 为 `terminated | truncated`，GAE 不跨 reset 传播。

## 统计与 checkpoint

`Curriculum/terrain_progress/*` 记录六类累计结果、成功率、实际升级/降级次数、平均等级及 L0–L9 环境比例。成功率为 `success/(success+failure)`，中性截断单独统计。完整计数按 **[出生 family, 出生 level, outcome]** 保存；L9 成功计入成功次数但不增加实际升级次数，L0 失败同理。

原生 `OnPolicyRunner.save()` 自动写入 checkpoint 的 `environment_state_dict`：

- 各环境的出生等级、地形类型，以及课程累计计数和实际升降级计数。
- 动作 sampler 的失败/成功 EMA、尚未合入 EMA 的计数和更新间隔进度。
- 校验信息：课程 schema、地形资产哈希、动作顺序/长度、动作 manifest 哈希、兼容表哈希及采样/reset 配置。

正常续训调用保持不变：

```python
runner.load(checkpoint_path)  # 默认恢复模型、优化器、iteration 和课程/采样状态
```

恢复要求环境数量、地形和动作配置一致，不兼容时明确报错。加载后按恢复的等级重建出生位置，再重新初始化机器人；保存时中断的回合不结算。**恢复的是训练进度，不是仿真现场或逐步相同的随机轨迹**：不恢复旧动作执行到哪一帧、刚体物理状态或 RNG，新的回合重新采样并从第 0 帧开始。

评测常用的 `runner.load(path, load_optimizer=False)` 默认不恢复课程；也可显式使用 `load_environment=False` 只加载网络侧状态。反之，`load_environment=True` 可在不加载优化器时显式恢复课程。没有环境状态的旧 checkpoint 会提示并使用新课程状态；从 stage 1 迁移到 stage 2 的权重映射、新优化器和 iteration 0 仍按 P0 处理，不能将其混同于同阶段续训。

当前验收覆盖单训练进程、相同环境数的恢复；完整 stage 2 的多进程状态合并和 RNG 恢复不在本接口中。

## 验证

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_curriculum.py tests/test_pgmt_rl.py tests/test_pgmt_sampling.py \
  tests/test_pgmt_contact_reference.py tests/test_pgmt_terrain_query.py \
  tests/test_pgmt_reference.py tests/test_pgmt_mdp.py tests/test_pgmt_config.py

/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_curriculum.py \
  --headless --device cuda:0 --output logs/pgmt_curriculum
```

79 项测试通过。Isaac [验证报告](../logs/pgmt_curriculum/validation.json) 通过 **998 个控制步**：使用原始 499 帧动作，核对六类判据、失败优先、L0/L9 封顶、旧 reward 与新 origin 的顺序、sampler 单次结算、手动 reset，以及真实 runner checkpoint 保存/恢复和恢复后继续升级。验证脚本控制注入失败/越界/超时标志，验证的是生命周期和状态转换，不是策略的地形跟踪成功率。`audit_checkpoint.pt` 是验收模型，不是正式训练权重。

历史 `logs/pgmt_motion_end_audit/validation.json` 对应课程接入前的只读审查，其“课程未实现”结论已由本次验收更新。采样/接触接口的独立验证脚本会关闭课程，避免手动改帧和统计注入干扰课程判据。
