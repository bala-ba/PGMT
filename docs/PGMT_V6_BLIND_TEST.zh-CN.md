# PGMT v6 新地形盲走训练测试

2026-10-11：已接入 Isaac GUI 训练并完成 1000 次 PPO 更新，使用 `assets/terrain/pgmt_stage2_v6`。这是沿用第一阶段策略和奖励的地形盲走测试，尚未加入 P2 感知与完整第二阶段奖励。

## 本次运行

- 有效运行目录：`logs/pgmt_v6_blind/20261011_02`；历史启动 PID：`2062651`。`training_status.json` 已记录 `completed`，最终更新时间为 2026-10-11 13:34:26；终端输出见 `console.log`。
- 初始权重：`logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt`，加载 iteration 15000，同时恢复 optimizer 并同步 PPO 学习率。
- 160 个环境，5 类各 32 个；初始课程等级 L0–L1，追加 1000 次 PPO 更新，每 50 次保存。测试 checkpoint 写入独立目录。
- 观测仍为 `proprio / proprio_history / command_window / critic`，移除 height scanner；49 个权重张量与源 checkpoint 完全相同，策略签名和观测 schema hash 一致。
- 862 段动作；动作级 adaptive sampling，`enable_frame_sampling=False`；从第 0 帧开始，脚部有效支撑起始帧限制保持关闭。
- GUI 默认看 `stairs / L0` 区域，显示全部机器人且相机不跟随。原有全局、固定地形区域、单机器人跟随按钮均保留，地形列表适配 5 类。

## 地形与 reset 逻辑

1. 导入原生地形 USD，校验 USD 与 800 个 PLY 的 SHA256；同一组 PLY 三角形构成 GPU 高度/距离查询网格。课程使用 50 个“样式 × 难度”区域，每区域 24×24 m、含 16 个 6×6 m 小块。
2. 兼容表仍使用现有 CSV。v6 的 `flat / stairs / slope / boxes / rough` 分别映射旧表的 `flat / up_stairs / up_slope / boxes / random_rough`；旧语义组 `stairs / slopes` 同样排除对应倒金字塔，显式 inverted 排除项也保留效力。
3. 在出生分配的区域内均匀抽取一个小块，再在距其边界 0.25 m 的内部均匀抽取 root XY，yaw 从 `[-π,π]` 抽样。全身碰撞包络须落入小块并通过真实网格净空检查；高度可为负。放置只改变统一平移/yaw，不修改源关节轨迹。
4. 失败仅重试位置/朝向，最多 128 次，不重新抽动作、不记 sampler outcome。放置通过后整个 episode 共用同一参考变换；未来窗口允许跨小块/区域，但必须留在有效地图。
5. 地形查询按当前世界位置解析区域和 `patch_id`。课程仍归属于出生区域；跨小块不 reset、不移动参考、不更新课程。超出有效地图显式 truncation。
6. 原有旧奖励结算、sampler 单次记账、动作耗尽截断、成功升级/失败降级以及课程保存恢复保持生效。v6 checkpoint 身份额外记录随机放置配置，防止以不同 reset 规则静默续训。

候选位置均匀抽样；最终通过全身检查的位置分布会受到动作姿态、地形边缘和未来窗口约束，并不保证严格均匀。

## 已验证与问题处理

| 项目 | 结果 |
| --- | --- |
| 相关单元测试 | 70 项通过，包含新旧地形查询、800 小块索引、随机位置覆盖、兼容性过滤、参考变换、接触、课程和配置 |
| 本次训练运行时网格/PhysX 检查 | 65,260 点，覆盖 800 小块、全部 5 类×10 级及连接处 |
| PhysX 对导出高度最大误差 | `2.471730113e-6 m` |
| GPU 查询对导出高度最大误差 | `3.814697266e-6 m` |
| 启动后前 10 次 PPO 更新 | loss/观测/奖励/参数有限，策略参数已实际更新 |
| 前 10 次更新的生命周期检查 | 240 控制步，205 次初始/reset 姿态检查，45 次 done 对应 45 次 sampler outcome，38,355 次存活环境参考不漂移检查，38,400 次接触帧新鲜度检查 |
| 随机放置 | 初始位置、朝向、小块、净空和重试次数保存于 `initial_spawns.json`；前 10 次更新最多重试 4 次 |
| 可视化 | 已检查 `training_viewport_1.png`，机器人分布在新地形的小块内部；后续另存第 5、50 次更新截图 |
| 旧 v5 保留 | 原配置、生成器、构建/预览脚本、manifest 和 USD 共 6 个文件 SHA256 均与 v6 构建时记录相同 |

首次试跑 `20261011_01` 触发 PhysX `collisionStackSize buffer overflow`，需求约 90 MB，默认仅 64 MiB，导致接触被丢弃。已停止并标记 `stopped_invalid`，其训练产物不作为有效结果。v6 启动入口将碰撞栈增至 **256 MiB**，有效试跑 `20261011_02` 重新加载原始第一阶段权重；截至上述检查未再出现碰撞栈溢出或丢弃接触。

启动阶段后续确认至 **30 次更新 / 720 控制步**：累计 410 次初始/reset 姿态检查，250 次 done 与 250 次 sampler outcome 一一对应；课程结果为 107 次完成、135 次失败、8 次地图越界，已出现 L2 出生分配。最大放置重试次数为 5。该时刻快照保存在运行目录的 `startup_validation.json`，源 checkpoint SHA256 再次确认未变；最终完成结果见下节。

加载第一阶段 checkpoint 时出现“没有 terrain curriculum state”提示是预期行为：本次新建地形课程，不沿用旧地形等级。已有 v6 checkpoint 则携带课程与 sampler 状态。本文启动测试不代表所有高难度动作已学会，也未替代所有十级随机姿态的逐项验收。

## 版本归档时的完成结果

`pgmt-p1-2026.10.11` 整理时读取已有最终状态，并核对实际保存的 checkpoint：

| 项目 | 最终记录 |
| --- | --- |
| 状态与更新时间 | `completed`；2026-10-11 13:34:26（Asia/Shanghai） |
| PPO 更新 / 控制步 | 1000 / 24,000 |
| 初始及 reset 姿态检查 | 10,582 次，包含首次初始化的 160 个环境 |
| done / sampler outcome | 均为 10,422 次 |
| 课程结果 | 5463 次完整跟踪、4683 次失败、276 次地图越界 |
| 参考变换 / 接触索引检查 | 3,829,578 / 3,840,000 次 |
| 最大放置重试 | 7 次 |
| 最终 checkpoint | `model_15999.pt`，实际 `iter=15999`，包含 optimizer、policy metadata、curriculum 和 sampler |

本次归档重新执行全部 PGMT 测试，**118 项通过**；同时校验 v6 配置、USD、800 个 PLY、生成器源码哈希及 v5 六个保留文件，源 Stage 1 权重也核对原 SHA256。结果摘要和文件哈希保存在 [版本快照](versions/pgmt-p1-2026.10.11.json)，对应说明见 [CHANGELOG](../CHANGELOG.md)。这些训练计数来自已有运行，归档整理未重启训练或 GUI。完成 1000 次更新不代表策略已在所有高难度地形收敛。

**后续补验已完成：** v6 全库随机出生、原生扫描、实际续训恢复和常用训练入口均通过。发现并修复已有课程统计被入口拒绝、normalizer 二次加载失败两处问题；补充后的回归测试为 119 项。见 [v6 P1 补充验收](PGMT_V6_P1_ACCEPTANCE.zh-CN.md)。新报告与修复独立于上述版本归档，不改写原 1000 次训练记录。

## 复现命令

在 PMT 根目录执行；默认创建新的时间戳日志目录：

```bash
bash scripts/train_pgmt_p1_gui.sh \
  --terrain_asset assets/terrain/pgmt_stage2_v6 \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --num_envs 160 --max_iterations 1000 \
  --viewer_mode tile --viewer_terrain stairs --viewer_level 0 \
  --viewer_tile_offset 14 -18 20
```

本次实际启动另外指定了 `--log_dir logs/pgmt_v6_blind/20261011_02`。不要复用该目录；入口会拒绝覆盖已有运行记录。去掉 `--terrain_asset` 仍按原 v5 地形和中心附近位置扰动启动。

## 代码入口

- `pmt_tasks/pgmt_terrain_layout.py`：v6 区域视图、旧 CSV 映射、小块完整性校验与随机 XY。
- `pmt_tasks/pgmt_terrain_importer.py`、`pgmt_terrain_query.py`：50 区域课程原点、800 小块碰撞查询、当前坐标的小块索引。
- `pmt_tasks/mdp/commands/pgmt_motion_command.py`：自动区分旧地形与新地形 reset，统一参考变换。
- `pmt_tasks/pgmt_curriculum.py`：v6 随机放置参数纳入恢复身份。
- `scripts/train_pgmt_p1.py`：可选 terrain asset、碰撞容量、动态 GUI、运行时检查及记录。
