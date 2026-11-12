# PGMT P2.4 / P2.5 实施与验收

日期：2026-10-13。执行计划先写入 [P2 计划 §4.0](PGMT_P2_PLAN.zh-CN.md)，再按「显式迁移 → 严格续训 → 实际 Isaac 短训 → 成本对照」执行。**P2.4 完成；P2.5 的 160 环境集成验收通过。** 1024 环境已实际尝试，受 PhysX 初始化显存限制，未通过扩容验收；其失效数据不计入结果。机器报告状态为 `passed_160_scope`。

后续定位补充：27.04 GiB 分配源于 Cloner 开启包围盒环境 ID 编码后，804 个共享地形 mesh 与所有机器人形状形成候选对；精确请求为 `1024×29×804×1216 bytes`。独立进程只把 `physxScene:envIdInBoundsBitCount` 改为 0 后，1024 环境完成初始化/reset/3 步，总显存快照 11,049 MiB。详见 [显存定位与对照](PGMT_PHYSX_MEMORY_DIAGNOSIS.zh-CN.md)。此为当时仅诊断的记录；后续修复已正式接入，1024 PPO/恢复及 8192 环境训练见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。本报告原有 PPO 验收范围仍为 160 环境。

## 实现及训练逻辑

- 正式任务 `pgmt_stage2` / `PMT-PGMT-Stage2-Perception-G1-v0`，对应 [task 配置](../configs/task/pgmt_stage2.yaml)、[环境](../pmt_tasks/env_cfgs/pgmt/stage2.py)、[agent](../pmt_tasks/agent_cfgs/pgmt.py)。专用入口是 [train_pgmt_stage2.py](../scripts/train_pgmt_stage2.py) 和 [启动脚本](../scripts/train_pgmt_stage2.sh)。使用此入口明确选择初始化/续训；旧通用 `train.py` 没有这套迁移参数和实际环境契约绑定。
- 默认从指定 `model_15000.pt` 初始化。严格检查旧签名/动作/value 分组/观测 schema，将旧网络、三头 critic、std、normalizer 的 **49 项状态**全部复制；只保留新建的 **10 项感知状态**。新 Adam、已完成更新数 0、全新 v6 sampler/课程。源文件 SHA256：`40708024acb51c6ce3cd87d6a067f59ff822763b6d8b73d3e2473d9953793087`。
- [PGMTStage2Runner](../motion_tracking_rl/runners/pgmt_runner.py) 包装原生 rollout/PPO；[迁移与恢复工具](../motion_tracking_rl/utils/pgmt_checkpoint.py)负责清单和严格校验。没有使用宽泛 `strict=False`，没有加载 Stage1 的 optimizer 或 iteration。
- `--resume` 仅接受 `kind=pgmt_perception_training, version=1`。在修改 live 状态之前检查实际训练配置、网络/观测/感知契约、tensor、Adam 参数顺序/槽位/超参数、LR、进度、RNG，以及 sampler/课程身份。恢复四组 sampler 计数及 EMA 时钟、各环境地形等级/类型和课程统计；随后按已恢复 RNG 重开物理回合，不结算中断回合。不承诺重现被中断的 PhysX 轨迹。
- checkpoint 的 `iter == next_iteration == 已完成更新次数`，续训从下一次更新开始。原生自动文件名仍采用循环的零基编号：例如 `model_24.pt` 内部是已完成 25 次；建议使用明确的 `final.pt` 和内部 `next_iteration`。断言通过后使用临时文件原子替换目标 checkpoint。
- 地形为 v6 五类、十级、每区域 16 个 6×6 m 小块；160 环境每类 32 个，训练初始 L0–L1。随机出生 XY/yaw，第 0 帧、无有效支撑门槛、动作级 adaptive sampling、`enable_frame_sampling=False`、`chunk_length=0`，保留兼容表过滤与独立地形课程。
- actor/critic 共用相同已保存增强地图和掩膜、独立前向，13 个 K/V；21×21、2×2 m，PMT version=3 高度/增强/归一化规则保持。新旧网络均训练；没有 attention bias 日程。保留原三组奖励与三头 value，地形接触奖励/第四头留给 P3。
- rollout 24 步，PPO 5 epochs、4 minibatches、gamma=0.99、lambda=0.95、clip=0.2、entropy=0.005；配置初始 LR=1e-4，原生 adaptive KL 调整。保存后续实际 LR，恢复时同时恢复 PPO 标量与 Adam 各组 LR。

## 实际 Isaac 验收

运行 Isaac Sim 6.0.1 + 原有 Isaac Lab 2.3 兼容层，复用 audited USD、Fabric 和 256 MiB PhysX collision stack；使用 `cuda:1`，RTX 5090 32 GiB。所有运行有限结束，已确认没有遗留 Isaac/CUDA 训练进程。实际仿真使用 PyTorch 2.10.0、NumPy 2.3.1；CPU/CUDA 单元回归使用 pmt 环境 PyTorch 2.7.0。报告读取包含 NumPy RNG 的 checkpoint 时使用同一 isaacsim Python，避免旧 NumPy 环境无法反序列化 `numpy._core`。

1. **旧策略回归**：在相同实际环境观测与 normalizer 下，禁用 terrain 后 actor 与三头 value 和指定 Stage1 checkpoint 的最大误差均为 **0**。再启用感知进行真实 PPO。
2. **几何与边界**：L0/L5/L9 各检查 160 个初始化，起始 frame=0、兼容表与最小碰撞间隙通过。每个等级五类各取一张完整扫描，与独立 PhysX raycast 对照，共 **6,615 点**，最大误差 **4.71e-6 m**；批量扫描与网格高度查询最大误差 **5.73e-6 m**，负高度凹地保留。容差 1e-4 m。
3. **跨 tile**：把机器人从 flat/L9 移到 stairs/L5，扫描按实际位置命中后者；出生等级/样式及统一参考旋转/平移保持不变。地图边缘样例 123/441 个几何有效点，网络无效值保持 -1；完全越界时四个 patch 无效、策略有限，并在真实 `env.step()` 返回 `done + time_out`。受控探针结束后精确恢复原持久计数和课程，再开始训练。
4. **25 次 GUI PPO 更新**：600 个控制步、96,000 条 transition；**221 次 reset = 221 次 done = 221 次 sampler outcome**。检查 95,779 次存活环境的参考变换、96,000 次接触标签 motion/frame 索引，无漂移、旧接触缓存或重复结算；奖励组求和与总奖励一致。课程事件为成功 97、失败 113、地图退出 11，等级自然更新至 L0–L2。
5. **真实梯度**：逐更新 loss/参数/梯度均有限，selector、patch encoder、IFM 每轮均有非零梯度，最终最大参数变化：selector `0.003185`、patch encoder `0.002531`、IFM `0.003303`。固定 16 个 flat 环境观测的中心最大变化 `0.2867 m`；低/中/高几何探针及整批统计覆盖五类。有效 patch 比例与逐 head attention 写入 `progress.jsonl`，固定观测的前后中心变化另存 NPZ。
6. **严格恢复**：独立进程加载第 25 次 checkpoint，再更新 5 次保存第 30 次；第三个进程加载第 30 次并再次保存。模型/normalizer、Adam、实际 LR、sampler 四计数/时钟/派生概率、课程等级/统计、内部 iteration 逐项一致；恢复 reset 不产生额外 outcome。RNG 在 reset 前恢复，物理 episode 重新开始。

## GUI 与可审查材料

[GUI helper](../pmt_tasks/pgmt_perception_viewer.py)提供全局/解除跟随、选定环境跟随、按样式/难度固定 tile、相机偏移及关闭叠加。青色为已保存增强观测中的有效点，红色为无效点，黄色为四个中心，洋红为 0.4 m 采样跨度。无效点为了可见显示在 torso 高度；显示上的高度抬升不进入网络。窗口显示 patch 0 的完整 5×5 高度/掩膜、四个 patch 的 coverage 和四个 head 的 terrain attention；四个 patch 的完整高度/掩膜均保存在 NPZ 和离线图中。

诊断只读取 PPO 已有观测，不额外调用观测函数或重新采样增强；调试前向不更新 normalizer、不保存训练图。几何受控检查和 GUI 检查明确独立于正式 rollout。

- [GUI 叠加截图](../logs/pgmt_p245/gui_160/training_viewport_5.png)
- [全部机器人与自由相机](../logs/pgmt_p245/gui_160/global_viewport.png)
- [stairs/L5 固定 tile](../logs/pgmt_p245/gui_160/tile_stairs_L5.png)
- [三个难度与五类地图/crop](../logs/pgmt_p245/geometry_glimpses.png)、[patch 高度/掩膜](../logs/pgmt_p245/patch_heights_masks.png)、[各 head attention](../logs/pgmt_p245/attention_heads.png)
- [机器报告](../logs/pgmt_p245/report.json)、[训练状态](../logs/pgmt_p245/gui_160/training_status.json)、[逐轮日志](../logs/pgmt_p245/gui_160/progress.jsonl)、[几何报告](../logs/pgmt_p245/gui_160/geometry_report.json)、[迁移清单](../logs/pgmt_p245/gui_160/migration.json)

## 问题与处理

| 发现 | 处理与证据 |
| --- | --- |
| Isaac `@configclass` 将默认字典移至实例，继承类定义时直接访问基类 `obs_groups` 报错 | Stage2 显式声明六组观测，实际 AppLauncher 环境下重新验证。失败启动保留在 `gui_160_a1.log` |
| 初次 GUI 曲线只有一个 widths 值，Hydra 按 vertex 插值，边框未显示 | 给 20 个边框顶点设置 20 项 widths 并指定 vertex 插值，最终截图中边框清晰可见 |
| 第一轮已经完成 25 次 PPO，但收尾断言误用 IFM 参数名 `ifm` | 改为真实参数前缀 `intent_attention`；保留 `gui_160_a2` 和错误日志，最终完整复跑通过的目录是 `gui_160` |
| 原生 runner 的 current iteration 是本轮零基编号、普通 loader 对不匹配允许降级 | PGMT 独立 runner 维护 completed/next iteration，显式迁移与严格续训分开；错误契约或损坏状态先拒绝再修改 |
| 第一次 PPO 更新 KL 较大、adaptive LR 降至 1e-7 | 同样的 blind 控制也出现首轮 KL≈0.178，完整感知 GUI≈0.167，不能归因于感知注入。scan_only 控制首轮约 0.825；GUI 随后 24 次 KL 在 0.00716–0.02509，末轮 LR=3.8443e-6。保持既定新 Adam 和原生 KL 调节，未引入额外 attention 退火；这不是首轮稳定性的保证，正式长期训练需单独调优初始 LR 并观察 KL/clip fraction |

固定 flat 探针末态的中心对重叠比例 93.75%、平均重叠面积占单 patch 的 37.97%，近边界（绝对坐标 >0.76 m）比例 0；25 轮整批有效 patch 比例均为 100%，越界探针四个 patch 则全部无效。短训中心相互重叠较多，未触及 ±0.8 m 的边界；这符合当前验收观察，不表示已学会分散寻找落脚点。P2 的通过范围是感知路径可训练、可保存恢复、可可视化；地形适应效果及完整论文训练需要 P3/P4。

## 吞吐、显存与回归

**321 项 CPU/CUDA 回归通过、0 skipped**，见 [JUnit](../logs/pgmt_p245_final_regression.xml) 和 [pytest 日志](../logs/pgmt_p245/pytest.log)。覆盖迁移、损坏/错配拒绝、RNG/reset、IFM/PPO、感知预处理与独立进程错误检测。最后新增监视器后再次完成 160 环境真实续训 `30→31`，见 [最终运行状态](../logs/pgmt_p245/guard_smoke_160/training_status.json)。

比较 headless、同 GPU、同 rollout/PPO 配置下的 blind（无扫描）、scan_only（扫描/存储地图但策略不用）、perception（完整融合）。每项 4 次更新，剔除首轮，后三轮取均值。PyTorch allocated/reserved 只计 Torch allocator，不能代表 PhysX/RTX 总显存；各运行另存 `nvidia-smi` 前后快照。GUI 主验收包含绘制和断言，不能作为纯训练吞吐基线。

| 环境数 | 模式 | rollout 采集 s | PPO 更新 s | env-step/s | Torch 峰值 MiB | GPU 总占用快照 MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 160 | blind | 3.916 | 0.126 | 949.9 | 1358.3 | 16151 |
| 160 | scan_only | 4.022 | 0.134 | 923.9 | 1380.6 | 16177 |
| 160 | perception | 4.057 | 0.184 | 905.5 | 1442.2 | 16249 |

完整感知相对 blind：本次采集耗时约 +3.6%，PPO 更新约 +45.4%，Torch 峰值约 +83.9 MiB，总吞吐约 -4.7%。只有后三轮均值，轨迹并非逐位相同，不能据此外推长期/独占性能；本轮各成本运行启动时 GPU 利用率为 0%，GPU 总显存也包含 RTX/PhysX 和显示占用。GUI 含绘制/审计时均值为采集 4.978 s、更新 0.185 s。

**1024 扩容未通过。** v6/800 个 collider 在首次碰撞初始化申请额外 `29,032,710,144 bytes`（约 27.04 GiB）失败；第一轮尝试时该设备启动前快照已用 4,385 MiB，总容量 32,607 MiB。即使无感知也不能在当前配置下正常启动，故没有继续 scan_only/perception 的 1024 对照，不提供伪吞吐。没有通过删除地形/碰撞、缩小地图或改变训练出生逻辑绕过。后续规模优化需另行检查初始碰撞分配与原生 mesh 的物理显存，不能仅根据网络张量显存决定 env 数。

原生错误未自动抛出 Python 异常，首次尝试的状态曾误报 completed；现已标记 `invalid_physics`，该目录下 checkpoint/性能全部作废并保留 `INVALID.txt`。最终增加 [独立进程监视器](../scripts/pgmt_process_guard.py)，遇到 PhysX Error/overflow 即终止自己启动的 worker，写入 failed_physics；超时 5 s 后强制结束，避免 CUDA 场景损坏时退出卡死。进程内 Python 日志回调曾卡住，已移除。真实 OOM 复现验证 **非零退出、0 PPO 更新、无 checkpoint**，见 [容量与保护结果](../logs/pgmt_p245/scale_limits.json)。监视器只处理失败运行，不修改正常环境/PPO 数据。

## 复现命令

默认新建输出目录；不能覆盖已有验收目录。以下命令从项目根目录执行。

```bash
# 指定 Stage1 -> P2；显示 GUI；默认 v6、160 环境、25 次更新。
bash scripts/train_pgmt_stage2.sh --acceptance

# P2 同配置严格续训；max_iterations 是额外更新数。
bash scripts/train_pgmt_stage2.sh --headless --num_envs 160 --max_iterations 5 \
  --resume logs/pgmt_p245/gui_160/final.pt

# 全流程：GUI 25 次 + 新进程续训 5 次 + 再次恢复 + 160 三组成本；1024 在 PhysX 容量允许时继续。
bash scripts/validate_pgmt_stage2.sh
```

长期训练可显式增大 `--max_iterations`；环境数目前只对 160 完成实际集成验收，1024 扩容需先解决上述物理显存问题。严格续训要求环境数、设备与训练/感知契约一致；改变规模应使用明确的迁移方案，不能把不匹配 checkpoint 当同配置续训。`--initialize-from` 可显式指定兼容的 P1/Stage1 盲策略，只迁移网络，sampler/课程仍重建；不会隐式带入 P1 的课程。
