# PGMT P4 数值验证、短训与正式训练

开始日期：2026-10-14；状态更新：2026-10-15。执行根目录：`logs/pgmt_p4_20261014/`。**P4 数值、有限短训及 1024 扩容/恢复验收通过；cuda:0、8192 环境正式训练已完成 383 次 PPO 更新，并按用户停止请求保存退出。** 监督状态为 `completed`、`stop_reason=stop_file`，结束时间为 **2026-10-15 10:18:08（北京时间）**；没有触发 OOM 降档，也未跑满原定 12 小时或 20,000 次更新预算。

正式 run：`logs/pgmt_p4_20261014/formal_8192/`；最终权重为该目录的 `final.pt`，`iter=next_iteration=383`，共 **9,192 个控制步、75,300,864 transitions**。本次版本整理已独立重读，模型/优化器张量有限，SHA256 为 `a10c42292b9023e9a786ca3336754fd1aaee6716c91411fecf60c14d3206c62d`。训练记录含 209,186 次 done、课程结果 92,461 次成功 / 111,472 次失败 / 5,253 次地图越界；这是训练窗口统计，不是固定评测完成率。全程平均 collection 99.25 s、PPO 更新 2.43 s。停止证据见 [stop_verification.json](../logs/pgmt_p4_20261014/stop_verification.json)，关键摘要和哈希已保存至 [版本快照](versions/pgmt-p4-2026.10.15.json)。

启动时的前 3 次更新及显存交接证据保留于 [HANDOFF.json](../logs/pgmt_p4_20261014/HANDOFF.json)。当前 449 个 Python/shell/YAML 文件与 `source_snapshot_formal.zip` 中的训练快照逐字节一致。随后独立做过两次采样耗时诊断，其 next_iteration=384 的权重不替代正式 383 版本，也没有恢复正式训练；详见 [耗时诊断](PGMT_COLLECTION_PERFORMANCE.zh-CN.md)。

## 当前约定

- 正式任务 `pgmt_stage2_terrain`，v6 五类十级、862 个动作，固定第 0 帧、动作级 adaptive sampling、兼容表过滤、随机 XY/yaw、四头和完整 27 项奖励。
- 指定初始化权重：`logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt`。有限短训和正式训练分开建目录；正式训练从该 Stage1 权重迁移，不沿用有限测试中人为几何探针的状态。
- 单卡 `cuda:0`。优先 8192 环境，发生 OOM 后依次尝试 4096、2048、1024；不同环境数量分别建新 run，从 Stage1 重新初始化，不把不同环境数量伪装成严格续训。
- 默认正式训练约 12 小时，或先达到 20,000 次 PPO 更新。时间预算在完整更新边界退出并保存 `final.pt`；不是 12 小时完成论文性能的承诺。有限验收不计入这段正式训练预算。
- Headless，保持正常地形课程和训练随机化。recovery pool/奖励仍关闭；保留当前身体惩罚及控制步末采样定义，已知短冲击峰值遗漏和爬行膝/手接触语义问题见 [身体碰撞验收](PGMT_BODY_CONTACT_AUDIT.zh-CN.md)。
- 本次范围是 P4 数值、有限训练、扩容与正式训练启动/运行。30 s 固定评测、MuJoCo 导出和论文性能对比另行验收，不把短训通过等同这些任务完成。

## 有限验证顺序

1. 当前数值回归：623 项 CPU/CUDA 通过，覆盖奖励、地图/梯度、参考变换、容差、接触对齐、四头 GAE/timeout、reset/cache、Stage1 回归及 checkpoint。
2. 新增日志/四头/续训补充回归：44 项通过（与前项有重叠）；PhysX 工厂作用域/失败恢复/OOM 分类 4 项通过。
3. PhysX 实体验证：5 环境/32 控制步通过。两台机器人完全重叠、空中自由运动，接触力和相对位姿差均为 0；flat 出生机器人进入 stairs/slope/boxes/rough 后可读取真实地形承重，固定参考和出生归属不变。
4. 平地迁移、五类地形短训与扩容结果如下，全部正常退出；独立重读磁盘 checkpoint 的 [finite_acceptance.json](../logs/pgmt_p4_20261014/finite_acceptance.json) 为 `passed`。

| 运行目录 | 环境数 | 本次更新 | 最终 next_iteration | 结果 |
| --- | ---: | ---: | ---: | --- |
| `flat_160` | 160 | 3 | 3 | Stage1→四头平地迁移、PPO 与日志通过 |
| `short_160` | 160 | 25 | 25 | 五类地形，含几何/跨 tile/截断探针 |
| `resume_160` | 160 | 5 | 30 | 新进程严格恢复并更新 |
| `restore_160` | 160 | 0 | 30 | 再次新进程，持久字段前后精确一致 |
| `scale_1024` | 1024 | 3 | 3 | 正式完整 PPO 扩容通过 |
| `resume_1024` | 1024 | 1 | 4 | 恢复后由时间限制触发，在完整更新边界保存退出 |

`short_160` 的 600 个控制步（96,000 transitions）产生 62 次完整动作成功、199 次失败、5 次地图越界；266 次 done 对应 266 次 sampler outcome，参考保持和接触刷新检查通过；出生课程达到 L2。短训结果仅证明执行链路和初步课程行为，不代表高难度完成率已验收。

1024 环境前三次更新的后两次平均采样约 25.96 s、更新约 0.88 s；对应恢复更新采样约 22.34 s、更新约 1.00 s。运行期间同卡还存在其他用户的进程，吞吐/显存以本机实测为准，未停止或更改其他进程。

有限验收早期新增 TensorBoard 标签使用 completed_iteration，比原生 Loss 横轴偏 1；正式版本已统一为原生零起始 iteration，`resume_1024` 实测对齐通过。早期各目录保留原始事件，不改写历史记录。

## 本次正式入口改动

### 共享地形 PhysX 内存修复

[pgmt_physx.py](../pmt_tasks/pgmt_physx.py) 仅在环境构造期间替换原生 scene 工厂，在 Cloner 完成之后、首次物理 reset 之前，将 `physxScene:envIdInBoundsBitCount` 从 4 设为 0，并读回确认。`replicate_physics=True` 和 `filter_collisions=True` 保留，安装的 Isaac 文件没有改动。作用域结束或构造失败都会恢复原工厂。

正式 [train_pgmt_stage2.py](../scripts/train_pgmt_stage2.py) 默认启用，参数为 `--env_id_bits 0`；该设置和 `--flat_only` 都进入严格训练契约。旧训练入口生成、未包含这两项契约的 Stage2 checkpoint 不能直接按“同配置”严格续训；本次新 run 的保存/恢复会独立验收。Stage1 显式迁移不受此限制。

原大额分配原因及单变量证据见 [PhysX 诊断](PGMT_PHYSX_MEMORY_DIAGNOSIS.zh-CN.md)。当前修复的正式训练规模仍以运行结果为准。

### 日志与安全停止

[pgmt_training_metrics.py](../pmt_tasks/pgmt_training_metrics.py) 只读采集已结算奖励，在 step 前保存出生 family/level，避免终止后自动 reset 将旧奖励记到新的 tile。新标签包含：

| TensorBoard 标签 | 含义 |
| --- | --- |
| `RewardStep/*` | 27 项加权、已乘一次 dt 的平均每步贡献，以及 total |
| `RewardRate/*` | 对应加权 rate，除回控制 dt 便于对比 |
| `RewardGroupStep/*` | upper/lower/terrain/auxiliary 四组每步贡献 |
| `TerrainReward/<family>/L*/<group>` | 按本步开始时的出生分配归集；无样本的格子不伪造 0 分 |
| `TerminationCount/*`、`TerminationFraction/*` | 当前 rollout 的各项 done 命中数/环境 transition 数；同一次 done 可同时命中多项 |
| `Curriculum/*`、`OutcomeWindow/*` | 各 family/level 环境数、累计结果、当前窗口结果与成功比例；比例分母为所有已结算结果 |
| `TrackingSnapshot/*` | 更新后当前状态的 upper/lower 位置 RMSE、姿态误差、关节 RMSE；不是终止前奖励快照 |
| `Sampler/*` | 动作采样概率直方图、熵、有效动作数量、最大/最小概率 |
| `Loss/*` | 原生 PPO 总/四头 value loss、surrogate、entropy、KL、clip fraction、LR 等 |
| `Perception/*` | 有效 patch 比例、覆盖率、选点边界/重叠、各头 terrain attention；最多约 256 环境的诊断采样 |
| `Perf/*`、`Memory/*` | 吞吐、采样/更新耗时、Torch 显存峰值；整卡显存另见 `gpu_monitor.jsonl` |
| `Run/*` | Text 页面中的实际启动参数和奖励权重 |

原生 `Episode_Reward/*` 和命令跟踪指标保留；它们与新增 `RewardStep/*` 的归一化/统计窗口不同，不应当成完全相同的量。

每个 run 保存 `params/env.yaml`、`params/agent.yaml`、`launch.json`、`migration.json`、`progress.jsonl`、`training_status.json`、TensorBoard events 和检查点。每 50 次更新保留一个周期 checkpoint，正常结束另存 `final.pt`；严格 checkpoint 含模型、归一化、Adam、LR、iteration、sampler 概率所需计数、地形课程和训练契约。

正式运行由 [run_pgmt_p4_long.py](../scripts/run_pgmt_p4_long.py) 在独立进程监督，记录 GPU 和执行日志；只对识别出的 OOM 自动降环境数量，数值/契约/其他错误保留现场并停止。达到预算后等待当前 PPO 更新完成；若停止请求 10 分钟后仍无响应，结束该运行所属进程组，并明确记录只能使用最近周期 checkpoint。

## 查看记录与停止新运行

本次训练记录的 TensorBoard 地址为 **http://127.0.0.1:6006**，日志根目录为本次执行根目录。停止验收时保留该服务；这里只记录其启动设置，不据此断言服务当前仍在线。服务仅绑定本机，已移除该服务进程的代理变量并使用 `--load_fast=false`，避免本地连接经系统代理失败。远程访问可通过 SSH 端口转发到本机 6006，不开放公共监听地址。

优先查看：执行日志、`overnight_status.json` 中的实际环境数/运行目录/状态、该 run 的 `training_status.json` 与 `progress.jsonl`。`formal_8192` 目录存在不代表该规模通过；若有 OOM，实际稳定运行会出现在下一档目录。

本次正式 run 已停止。对于使用同一监督器的新运行，请在其执行根目录创建停止文件，请求在下一个更新边界保存并停止；以下为本次运行使用的路径（文件已存在）：

```bash
touch /home/lzp/projects/perceptive_tracking/PMT/logs/pgmt_p4_20261014/STOP_REQUESTED
```

只有监督进程报告 `completed` 且对应 `final.pt` 存在，才表示正常保存完成。原生崩溃/强制停止时查看最后周期 checkpoint，不将 `initial.pt` 当成训练后的权重。续训必须使用相同环境数量和训练契约、新日志目录，并删除或改用旧的停止文件路径。
