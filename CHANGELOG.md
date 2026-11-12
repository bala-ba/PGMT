# PMT 修改与版本记录

使用带注释的 Git tag 记录研究阶段版本；标签固定对应的代码、配置和文档。Python 包的 `0.0.0` 占位版本保持原样。

## pgmt-p4-2026.10.19 — 训练优化、多 GPU 与持续支撑奖励

- 日期：2026-10-19（Asia/Shanghai）。
- 基线：[`e7e70fa`](https://github.com/bala-ba/PMT/commit/e7e70fab6dbfb8c215937a3261ceecbfc7504170)，版本 `pgmt-p4-2026.10.15`。
- 版本标签：[`pgmt-p4-2026.10.19`](https://github.com/bala-ba/PMT/tree/pgmt-p4-2026.10.19)。
- 源码哈希、回归结果、历史验收摘要及权重 SHA256：[版本快照](docs/versions/pgmt-p4-2026.10.19.json)。

### 本次修改

| 模块 | 当前行为 |
| --- | --- |
| T1 持续支撑质量 | 使用 `support_pattern_mean_v2`：双脚实际承重与同帧参考支撑模式全部有效且逐脚匹配时，持续结算参考支撑脚的平均几何质量；腾空、额外/缺失支撑为零，重新落脚没有事件奖金。未加权率上限 1，权重 10、dt=0.02 时总上限 0.2/步。结算方式进入严格恢复契约。 |
| 出生与超时 | Stage2 默认从每个 4×4 区域中央 2×2 小块采样 XY/yaw；保留全身净空、边界和未来参考检查。10 s 硬上限增加两个控制步的末帧容差，不提前截断未完成动作，课程完整动作仍计成功一次。 |
| Collection 优化 | 新增 reset 统计、索引缓冲清零、参考缓存/几何放置、接触传感器、延迟审计和奖励编译开关；原生 Sim 5.1 headless 入口默认 `--performance all`，执行选项独立写入 checkpoint。 |
| PPO 优化 | actor/critic 同批联合编码，保留双方梯度和原采样 RNG；日志在设备上累计、轮末集中读取，保留 KL 学习率判断及异常检查。 |
| 多 GPU | 支持 `--gpus`，总环境数均分到独立 Isaac 进程；同步梯度、KL 和归一化统计。rank 0 保存共享参数和所有 rank 的课程/sampler/RNG，更新边界协调停止，严格验证续训拓扑。本机分布式子进程可单独预加载兼容 NCCL。 |
| 回放与分析 | 新增 Stage1 平地、Stage2 五类地形策略回放及无物理参考机器人；支持暂停切换地形/难度、指定动作测试。新增训练指标与奖励分析脚本，整理旧 T1 的 20,000 更新训练分析和 GUI 启动说明。 |
| 文档整理 | 更新版本入口、训练命令和验证口径，修正旧 383/393、`train_01` 权重可直接按当前配置续训的过期示例，区分历史性能/奖励实验与当前配置。 |

### 本次验证与训练快照

- **CPU 回归 1037 项通过、87 项跳过、0 失败**：`pmt` 1012 通过/85 CUDA 或 NCCL 跳过；NumPy 2 的 `isaacsim` 环境四头专项 25 通过/2 CUDA 跳过。两进程 Gloo 训练/保存/恢复/停止通过。另 1 项既有 SONIC 用例因 Isaac 许可交互未执行；主套件排除的 27 项四头用例由第二套件覆盖。命令见 [使用指南](docs/USER_READ.md#当前版本回归检查)。
- 44 个变更 Python 文件通过语法解析，3 个新增 shell 脚本通过 `bash -n`，奖励 YAML 解析通过。本次隐藏 GPU 执行回归，未新启 Isaac/GUI 或正式训练。
- **旧 T1 的 `train_01` 已完成 20,000 更新**：8192 环境、每轮 24 控制步，共 3,932,160,000 transitions；2026-10-17 03:51:36 状态为 `completed / iteration_limit`。独立重读 `final.pt`，`iter=next_iteration=20000`，模型/Adam 有限，完整持久状态存在；SHA256 为 `e3f23fde8bbc44a82d98b64faa0d45c471e56eb088aae5f34b0bf3f2d1c81acd`。
- **新 T1 的 `train_02` 仍在运行**：2026-10-19 16:12:51 状态快照为 13,318 更新，8192 环境、单卡 cuda:1。另核验已经保存的 `model_13200.pt`：内部 `next_iteration=13201`，模型/Adam 有限，契约含 `support_pattern_mean_v2`。运行中状态与较早的已保存检查点分别记录；本次未停止、重启或更改该训练。
- 已核对既有有限验收原始摘要：T1 五类十级、67 控制步；超时 56 控制步/21 次 done 对应 21 次 outcome；中央出生、双 GPU 20 环境短训/恢复/协同停止；Stage2 五类地形 8 次切换、520 步回放。它们是历史验收，不计作本次重跑，报告后的源码变动在快照中另列。
- 历史共享编码 A/B 的 PPO 均时从 2.856 降至 1.507 s/轮；集中日志的单独收益约 1.1%，collection 编译单项收益仍无法从共享 GPU 的顺序短测中独立量化。详见 [collection](docs/PGMT_COLLECTION_OPTIMIZATION.zh-CN.md)、[共享编码](docs/PGMT_SHARED_ENCODING.zh-CN.md) 和 [日志](docs/PGMT_PPO_LOG_SCALARS.zh-CN.md) 报告；这些旧 T1 测量不代表新奖励的吞吐或策略质量。

### 恢复与范围

当前奖励、timeout、出生范围及分布式拓扑都受严格契约保护。383/393 旧权重与当前任务不同，`train_01` 也缺少新 T1 settlement，不能直接作为当前任务的同配置 `--resume`；旧 policy 可继续回放。原 Stage1 和 383 更新权重哈希未变。

新 T1 训练尚未结束；旧 T1 完成 20,000 更新不等于全地形收敛。recovery 按原约定暂缓，固定 30 s 评测、完整地形/难度成功率、导出和收敛仍未建立。权重、生成资产和原始日志保留本地；Git 保存源码、配置、测试、文档及关键证据摘要。

## pgmt-p4-2026.10.15 — 四头地形训练与 8192 环境运行记录

- 日期：2026-10-15（Asia/Shanghai）。
- 基线：[`6423d17`](https://github.com/bala-ba/PMT/commit/6423d172534907945c44f7a9b42ca39aa9a9b265)，版本 `pgmt-p2-2026.10.14`。
- 版本标签：[`pgmt-p4-2026.10.15`](https://github.com/bala-ba/PMT/tree/pgmt-p4-2026.10.15)。
- 源码、验收摘要和权重 SHA256：[版本快照](docs/versions/pgmt-p4-2026.10.15.json)。

### 本次修改

| 模块 | 当前行为 |
| --- | --- |
| 正式四头任务 | 新增 `pgmt_stage2_terrain`、`train_pgmt_terrain.sh` 和 27 项奖励配置，价值/奖励组为 `[upper, lower, terrain, auxiliary]`；原 P2 保留 21 项/三头。PPO 增加分头 value loss 日志，保留分头 GAE、优势求和后归一化。 |
| 接触与奖励 | 接入 T1 落脚质量、T2 参考接触匹配、T3 滑移、T4 绊碰、T5 快速切换、T6 力平方惩罚。地形过滤承重与全对象接触分别建模，缓存旧回合奖励快照，局部 reset 与重复读取不重复推进事件或计时。 |
| 跟踪与终止 | stairs/slope/boxes 上三项下肢容差随当前位置 torso tile 难度变化；Stage2 EE 加速度仅比较双腕。四项跟踪失败连续超限 0.2 s 才终止，回合 10 s；动作耗尽和地图越界立即截断。Stage1/P1 原行为保留。 |
| 迁移与恢复 | 支持 Stage1/P2 三头显式迁移，旧行映射到 0/1/3，新 terrain 行为零；新建 Adam、课程和 sampler。四头 checkpoint 使用独立 kind，严格校验并恢复完整持久训练状态。物理回合重新开始。 |
| PhysX 扩容 | 正式入口在 Cloner 后、首次物理 reset 前设置 `envIdInBoundsBitCount=0` 并读回；保留环境碰撞过滤，构造结束/失败均恢复原 scene 工厂。该设置与 flat-only 模式进入训练契约。 |
| 日志与运行管理 | 新增 27 项奖励、四组回报、地形等级、终止、课程、采样器、感知、吞吐和显存记录；监督器支持更新边界保存停止、时间预算及仅针对 OOM 的环境数降档。 |
| 验收与诊断 | 新增接触/奖励/容差/终止/四头/身体碰撞/物理隔离专项工具；归档 P3/P4 验收和 collection 耗时诊断，整理最新状态与训练命令。 |

### 验证与已保存训练

- 本次按依赖环境拆分回归：`pmt` 环境 **990 项通过**，NumPy 2 的 `isaacsim` 环境四头专项 **27 项通过**，合计 **1017 项、0 失败、0 跳过**；另 1 项 SONIC 测试因已知 Isaac 许可交互未执行。前一命令排除的另 27 项由后一命令完整覆盖，具体命令见 [使用指南](docs/USER_READ.md#当前版本回归检查)。59 个 Python 文件语法和 8 个 shell 脚本语法通过。
- 环境差异已记录：最初在 NumPy 1 全量执行为 1008 通过/9 失败，失败均为读取 NumPy 2 checkpoint；训练环境全量为 1015 通过/2 失败，失败为两个离线动作测试的 SciPy Euler 输入形状差异。最终按上述环境分别覆盖全部目标用例，未改写依赖、测试或 checkpoint 来掩盖差异；训练环境关闭有依赖冲突的第三方 pytest 自动插件。
- 核对已有 P4 有限验收：160 环境平地 3 次更新、五类地形 25+5 次更新及再次独立恢复通过；1024 环境 3 次更新、恢复追加 1 次并在时间边界保存通过。5 环境/32 控制步的机器人隔离与跨 tile 接触验证通过。
- **正式训练 `formal_8192`：cuda:0、8192 环境、383 次更新、9,192 个控制步、75,300,864 transitions。** 用户停止请求在更新边界生效，2026-10-15 10:18:08 监督状态记录为 `completed / stop_file`，无 OOM 降档；未跑满原定 12 小时或 20,000 次更新。
- 已独立重读正式 `final.pt`：`iter=next_iteration=383`，模型/Adam 有限，完整状态和 SHA256 与停止记录一致；当前 **449 个代码/配置文件**与正式训练源码快照逐字节一致。Stage1 源权重未变，详细证据见 [P4 执行记录](docs/PGMT_P4_EXECUTION.zh-CN.md)。
- 另有独立单轮耗时诊断：collection 80.63 s，其中 96 个物理子步 79.15 s，占约 98.16%；这是主机墙钟归因和单轮测量，不是 GPU kernel 独占时间或性能修复。诊断的 384 更新 checkpoint 不替代正式 383 更新权重。

本次整理未启动新的 Isaac/GUI 或正式训练。recovery pool/奖励按用户既定范围暂缓；短冲击末子步采样遗漏、爬行膝/手必要接触语义、分地形/难度策略表现、固定 30 s 评测、导出和收敛仍需分别验证。P3/P4 的训练链路可用不等于完整性能验收。生成资产、权重和原始日志保留本地，Git 保存实现、配置、测试、文档及关键摘要/哈希。

## pgmt-p2-2026.10.14 — PhysX 显存诊断与 P3 奖励审计

- 日期：2026-10-14（Asia/Shanghai）。
- 基线：[`88dadd0`](https://github.com/bala-ba/PMT/commit/88dadd0dc389e5f45d035ba2abab48e05637405f)，版本 `pgmt-p2-2026.10.13`。
- 版本标签：[`pgmt-p2-2026.10.14`](https://github.com/bala-ba/PMT/tree/pgmt-p2-2026.10.14)。
- 源码、诊断结果及 SHA256：[版本快照](docs/versions/pgmt-p2-2026.10.14.json)。

### 本次修改

- **PhysX 显存定位**：归档独立诊断脚本及四组对照。1024 环境原始请求满足 `1024×29×804×1216 = 29,032,710,144 bytes`，对应共享地形与机器人形状的候选对缓存；增加初始克隆间距没有改善，只在诊断进程将 `physxScene:envIdInBoundsBitCount` 从 4 改为 0 后显著减少候选对。详见 [诊断报告](docs/PGMT_PHYSX_MEMORY_DIAGNOSIS.zh-CN.md)。
- **可复现入口**：将本地启动封装整理为 `scripts/diagnose_pgmt_physx_memory.sh`，支持环境路径覆写、参数转发和独立输出目录；继续使用已有 PhysX 错误监视器。正式训练配置和训练入口保持上一版本行为。
- **P3 奖励审计**：归档当前 21 项/三头的实际计算路径，以及完整 28 项/四头的候选公式、权重、dt、掩码、事件历史和迁移边界。记录 T1 的脚—地形向上力承重判据、单次落脚触发及 0.20×0.10 m 的九点窗口；未确认的工程门槛继续标为候选。统一 T6 为文中最新无阈值力平方候选，修正旧的 350 N 阈值示例。详见 [奖励审计](docs/PGMT_P3_REWARD_AUDIT.zh-CN.md)。
- **文档整理**：同步 P2 验收补充、Stage 2 计划、README 和常用命令；将新增文档中的源码行号链接改为 GitHub 可访问的锚点。

### 验证与范围

- 本次运行 `python -m pytest -q tests/test_pgmt_mdp.py tests/test_pgmt_process_guard.py`：**8 项通过、0 项跳过**；诊断脚本 Python 语法、启动脚本 shell 语法和离线参数转发检查通过。
- 当前 21 项奖励权重与 YAML 一致；诊断 Python 源码哈希与已有报告一致；四组原始状态文件均为 `passed`，包围盒属性、候选对数量、PhysX 内存和整卡显存快照与汇总报告逐项一致。
- 已有诊断：160 原配置与仅增大间距时，初始化候选对均为 3,730,880；bits=0 的 160/1024 环境分别为 4,160/26,624。1024 对照完成初始化、reset 和 3 个控制步，第 3 步整卡显存快照为 11,049 MiB。
- **诊断结果不代表完整 PPO 扩容通过**：本次未启动 Isaac、训练或新的物理对照；正式 bits=0 修复尚未接入，环境间碰撞隔离、跨地块接触和 1024 PPO/保存恢复仍需验收。P3 奖励、四头网络和恢复机制尚未实现；上一版 160 环境训练验收范围保持不变。

生成资产、权重和原始日志保留本地；Git 保存诊断工具、审计文档及可追溯的结果摘要和哈希。

## pgmt-p2-2026.10.13 — Stage 2 感知融合与严格续训

- 日期：2026-10-13（Asia/Shanghai）。
- 基线：[`f22b03b`](https://github.com/bala-ba/PMT/commit/f22b03b16c5c49fa6d22b56c3624e59818d1b779)，版本 `pgmt-p1-2026.10.11`。
- 版本标签：[`pgmt-p2-2026.10.13`](https://github.com/bala-ba/PMT/tree/pgmt-p2-2026.10.13)。
- 源码、本地验收摘要及 SHA256：[版本快照](docs/versions/pgmt-p2-2026.10.13.json)。

### 本次修改

| 模块 | 修改内容 |
| --- | --- |
| 感知观测 | 实现 v3 契约：实际 torso yaw 坐标系、2×2 m / 21×21 高度图，`sensor_z - ground_z`、米制裁剪、每次观测调用独立噪声/dropout；高度图与有效掩码原子传递，归一化仅在网络进行。 |
| TerrainGlimpse / IFM | 从高度图学习四个 5×5 局部采样块，经共享编码器加入原有历史与未来指令，形成 13 个 attention KV；支持关闭感知、无效区域掩码和全无效回退。actor/critic 使用 rollout 保存的同一增强观测。 |
| 正式训练入口 | 新增 `pgmt_stage2` 配置、`PGMTStage2Runner` 和 `train_pgmt_stage2.py/.sh`；默认 v6 地形、160 环境、GUI、三组奖励/价值头。P1 入口继续保持无感知训练。 |
| 初始化与恢复 | `--initialize-from` 严格复制 Stage 1 的 49 个旧张量、初始化 10 个感知张量，重建 Adam/课程/采样状态并从 0 计数；`--resume` 预检契约后恢复模型、归一化器、Adam、学习率、课程、sampler、RNG 与完成更新数。物理 episode 重新 reset，不承诺逐步物理轨迹复现。 |
| GUI 与物理错误处理 | 显示选中机器人的高度图、采样中心/局部块及注意力诊断，支持跟随、全局和固定地块视角；独立进程监视 PhysX 错误，终止本次启动的 worker 并记录 `failed_physics`。 |
| P1 修复与验收 | 归一化器原位更新标准差 buffer，修复 inference mode 后加载；P1 续训允许恢复非零课程计数。补齐 v6 随机出生、扫描及真实 checkpoint 续训验收记录。 |

### 本次回归检查

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt*.py tests/test_normalization_resume.py \
  tests/test_all_tasks_resolve.py tests/test_builder_slice.py \
  tests/test_compat_matrix.py tests/test_group_yaml_valid.py \
  -k 'not test_sonic_agent_cfg_modes_resolve_release_onnx'
```

结果：**745 项通过、0 项跳过、1 项排除**，包含 CPU/CUDA；39 个变更 Python 文件通过语法解析，5 个 shell 脚本通过 `bash -n`，diff 检查通过。排除项为既有 SONIC 配置测试：当前环境导入 `isaaclab.utils` 会启动 Isaac EULA 交互，首次包含该项的检查因此失败；未把它记为通过。JUnit 的 34 条警告来自 `record_property` 与 xunit2 格式兼容性。

### 已有仿真验收与范围

- [P2.1 语义对齐](docs/PGMT_P21_PMT_ALIGNMENT.zh-CN.md)、[P2.2 TerrainGlimpse](docs/PGMT_P22_ACCEPTANCE.zh-CN.md)、[P2.3 融合](docs/PGMT_P23_ACCEPTANCE.zh-CN.md) 与 [P2.4/P2.5 集成记录](docs/PGMT_P245_ACCEPTANCE.zh-CN.md) 已归档；P2.4/P2.5 报告记录的 18 个核心实现文件哈希与当前源码一致。
- 集成报告状态为 **`passed_160_scope`**：160 环境 GUI 完成 25 次 PPO 更新、600 个控制步、96,000 次环境 transition；49 个迁移张量一致，关闭地形输入时 actor/critic 输出误差为 0。L0/L5/L9 初始化、跨地块、地图边界/全无效检查通过，6,615 个 PhysX 扫描点最大误差约 4.71 μm。
- 独立进程完成 25→30 次更新的续训与第 30 次更新的再次恢复；物理监视器接入后又完成 30→31 次更新。报告中的历史回归为 321 项通过、0 跳过，与本次 745 项检查分别记录。
- **1024 环境未通过**：blind 基线初始化时 PhysX 请求约 27 GiB 额外分配，发生 GPU OOM；该规模的 scan-only/perception profile 未运行。错误试跑已标为无效，监视器验证为非零退出、0 次更新、无 checkpoint。
- 原 v6 P1 的随机 XY/yaw、干净高度扫描和真实恢复专项验收已完成，详见 [P1 v6 验收](docs/PGMT_V6_P1_ACCEPTANCE.zh-CN.md)。

本次整理重新运行回归测试并核对已有报告，未启动 Isaac/GUI 训练。另一个 1000 次更新的用户运行按读取时状态单独保存在版本快照中，不计入上述集成验收。当前版本为 **P2 感知接入**；第四价值头、地形奖励、恢复机制及完整策略收敛评估仍在 P3/P4 计划中。地形/动作/机器人资产、checkpoint 和原始日志保留本地，Git 保存实现、配置、文档及关键哈希。训练与恢复命令见 [常用命令](docs/USER_READ.md)。

## pgmt-p1-2026.10.11 — v6 重复小块地形与随机出生

- 日期：2026-10-11（Asia/Shanghai）。
- 基线：[`a084b14`](https://github.com/bala-ba/PMT/commit/a084b142d6fc045835027c7ca296fa19c41f7c2e)，版本 `pgmt-p1-2026.10.10-r2`。
- 版本标签：[`pgmt-p1-2026.10.11`](https://github.com/bala-ba/PMT/tree/pgmt-p1-2026.10.11)。
- 本地记录摘要及 SHA256：[版本快照](docs/versions/pgmt-p1-2026.10.11.json)。

### 本次修改

| 模块 | 修改内容 |
| --- | --- |
| v6 地形 | 新增五种原生 Isaac Lab 地形：`flat / stairs / slope / boxes / rough`，十级、50 个 24×24 m 区域；每区域 4×4 个 6×6 m 小块，共 800 块。配置、几何构建、预览和 PhysX 检查独立于 v5。 |
| 几何参数 | 楼梯踏深固定 0.32 m；倒金字塔 slope 平台宽 0.5 m；boxes 采用原生正负高度扰动、去除中央平台，保留 0.4 m 小块连接带；rough 保留原生噪声和插值。 |
| 随机出生 | v6 自动按出生区域抽取小块、内部 XY 和 yaw，全身碰撞/净空/边界检查后接受；固定动作第 0 帧。最多重试 128 次，失败候选仅重新抽位置和朝向。 |
| 地图与兼容性 | importer 将 800 个碰撞小块映射为 50 个课程区域；按当前位置返回 `patch_id`，允许跨小块。五类样式映射到既有 CSV 的语义组和倒金字塔禁配规则。 |
| 课程与恢复身份 | 保留出生区域课程和统一参考变换，v6 的放置模式、边界、yaw、净空和重试上限加入 checkpoint 身份校验。 |
| 训练入口 | 新增 `--terrain_asset`；默认仍使用 v5。GUI 地形按钮按实际样式数量生成，记录小块分配、初始姿态及重试次数。v6 PhysX 碰撞栈提高至 256 MiB。 |

### 验证与训练记录

- 本次运行 `python -m pytest -q tests/test_pgmt*.py`：**118 项通过**，覆盖 v5/v6 查询、800 小块索引、跨缝、随机位置覆盖及兼容过滤等；Python 语法、shell 语法、离线启动参数和 diff 检查通过。
- v6 配置、USD、800 个 PLY 与 manifest 哈希一致；生成器源码与构建时哈希一致，v5 六个保留文件未变。
- 核对已有预览报告：78,060 次几何/候选表面/接缝查询通过，最大高度误差约 2.472 μm；训练启动另外完成 65,260 点 GPU/PhysX 查询。
- 有效运行 `logs/pgmt_v6_blind/20261011_02` 已于 **2026-10-11 13:34:26** 记录 `completed`：160 个环境、1000 次 PPO 更新、24,000 个控制步、10,422 次 done/sampler outcome；最终保存 **`model_15999.pt`**，包含课程、sampler 和优化器状态。源 Stage 1 权重哈希未变。
- 首次试跑 `20261011_01` 因碰撞栈溢出和接触丢弃标记为 `stopped_invalid`，未计入有效训练结果；增大容量后重新从 Stage 1 权重启动有效运行。

本次版本整理未重跑 Isaac/GUI 或 PPO。预览候选点不等于有效机器人姿态，训练完成不等于全地形收敛。v6 感知扫描及完整续训恢复尚需专项仿真验收，Terrain-Glimpse/IFM、地形奖励等完整 Stage 2 工作继续按原计划推进。

查看 [v6 地形说明](docs/PGMT_TERRAIN_V6_PREVIEW.zh-CN.md)、[盲走测试与完成结果](docs/PGMT_V6_BLIND_TEST.zh-CN.md) 和 [常用命令](docs/USER_READ.md)。资产、权重和原始日志仍保留在本地；新环境需准备 v5/机器人/动作资产，并生成 v6 manifest 后运行小块布局测试。版本快照保存关键结果及哈希，上一版记录保持独立。

## pgmt-p1-2026.10.10-r2 — 训练 GUI 视口控制

- 日期：2026-10-10（Asia/Shanghai）。
- 基线：[`274f1ff`](https://github.com/bala-ba/PMT/commit/274f1ff77944097e0bc0be188de43bcde33140ab)，版本 `pgmt-p1-2026.10.10`。
- 版本标签：[`pgmt-p1-2026.10.10-r2`](https://github.com/bala-ba/PMT/tree/pgmt-p1-2026.10.10-r2)。
- 本地记录摘要及 SHA256：[版本快照](docs/versions/pgmt-p1-2026.10.10-r2.json)。

### 本次修改

`scripts/train_pgmt_p1.py` 增加三种训练视口模式：全局自由视角显示全部机器人；单机器人跟随可按地形按钮选择；固定地块视角可查看七种地形的 L0–L9。全局/地块模式进入时定位一次，随后保留手动相机操作；地块视角使用资产原生 origin，保留倒金字塔负高度，不随机器人 reset 或课程变化移动。相机偏移可在面板调整，并检查有限值及非零长度。

| 参数 | 用途与默认值 |
| --- | --- |
| `--viewer_mode` | `follow` / `global` / `tile`，默认 `follow` |
| `--viewer_terrain` | 初始地块样式，默认 `flat` |
| `--viewer_level` | 初始查看等级 L0–L9，默认 0 |
| `--viewer_tile_offset DX DY DZ` | 相对地块 origin 的世界坐标偏移；默认根据地块尺寸设置 |

界面切换修改相机与机器人可见性；本次代码差异未修改 PPO、奖励、采样、课程或 reset 逻辑。README、[常用命令](docs/USER_READ.md) 和 [视口模式说明](docs/PGMT_P1_ACCEPTANCE.zh-CN.md#视口模式切换) 已同步。

### 验证与记录范围

- 本次重新运行 `python -m pytest -q tests/test_pgmt*.py`：**111 项通过**；Python 语法、GUI shell 脚本语法及 diff 空白检查通过。
- 从脚本的 argparse 定义离线解析了默认参数、全局模式和文档中的地块示例。直接运行 `--help` 在 Isaac 导入时遇到 EULA 交互，未完成完整入口检查；本次未启动 GUI/PPO，也未重新进行 GUI 按钮交互验收。
- 已有运行 `20261010_212841` 的状态文件最后更新于 **2026-10-10 22:03:05**，记录 **667 次 PPO 更新、iteration 15666**，视口为 `tile / L01_stairs`、140 个机器人可见。最后常规 checkpoint 为 `model_15650.pt`。快照保存这些记录及文件哈希；文件中的 `status=training` 仅为当时写入值，未据此断言当前进程仍在运行。
- 上一版 P1 验收与 390 次更新的停止记录保持独立。完整 Stage 2 的待办范围和本地资产依赖沿用上一版；生成数据、权重和原始日志继续保留在本地。

## pgmt-p1-2026.10.10 — P1 地形训练与验收

- 日期：2026-10-10（Asia/Shanghai）。
- 基线：[`7544eab`](https://github.com/bala-ba/PMT/commit/7544eaba24dc0b894ec0512a6490a0c25dc39aff)，STAGE 2 地形和机器人位置初始化。
- 版本标签：[`pgmt-p1-2026.10.10`](https://github.com/bala-ba/PMT/tree/pgmt-p1-2026.10.10)。
- 本地证据摘要及 SHA256：[版本快照](docs/versions/pgmt-p1-2026.10.10.json)。

### 本次修改

| 模块 | 当前行为与主要入口 |
| --- | --- |
| 动作采样与初始化 | 按出生地形读取人工兼容 CSV，再在允许动作中混合失败自适应和均匀采样；完整 segment、固定第 0 帧、关闭帧自适应及足部支撑过滤，保留爬行动作。见 `pgmt_sampling.py`、`pgmt_motion_command.py`。 |
| 训练配置归档 | 将 862 条动作的 `motion_terrain_compatibility.csv` 纳入 Git；NPZ 等生成数据继续忽略，人工筛选配置可随标签恢复。 |
| 接触参考 | 当前、未来 8 帧和局部环境共用 command 索引；提供左右脚接触、支撑/摆动及有效性；超出片段的未来标签无效，reset 后即时读取新标签。见 `pgmt_contact_reference.py`、`pgmt_runtime.py`。 |
| 地图与跨地块 | 按当前世界位置查询每只脚的地形和高度；允许跨 tile，出生课程分配保持独立；机器人或参考超出有效地图时显式截断。见 `pgmt_terrain_query.py`、`pgmt_terrain_importer.py`。 |
| 高度扫描 | torso 中心、yaw 对齐、2 m × 2 m、21 × 21 射线；输出带符号相对高度及有效掩码，共享全部 70 块地形网格，控制步/reset 刷新。见 `pgmt_height_scan.py`、`pgmt_height_scanner.py`。 |
| 地形课程 | 完整动作跟踪成功升级、失败降级，中性截断不改等级，L0–L9 封顶；先结算旧奖励，再更新出生位置、单次记账和 reset。见 `pgmt_curriculum.py`。 |
| 保存与续训 | runner/wrapper 保存恢复地形等级、课程与 sampler 统计，并校验资产/动作/配置；支持只加载网络的模式，旧 checkpoint 使用新课程状态。 |
| P1 无感知训练 | `PGMTBlindTerrainEnvCfg` 与 `train_pgmt_p1.py` 保持 Stage 1 四组输入和三头价值网络；严格核对权重/schema，恢复优化器学习率并记录实际 PPO 更新。 |
| 仿真与 GUI | 导入地形时停用内嵌 PhysicsScene，统一训练物理时钟；增加显式启用的 Isaac Sim 6 刚体兼容层和 Fabric GUI 入口 `train_pgmt_p1_gui.sh`。 |
| 验收工具 | 新增全库起始姿态、接触/采样、地图、高度扫描、课程/续训检查及 P1 验收编排脚本，补齐中文接口和使用文档。 |

### 验证与训练状态

本次整理重新执行以下检查：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt*.py
bash -n scripts/train_pgmt_p1_gui.sh
git diff --check
```

结果：**111 项测试通过**；改动 Python 文件全部通过语法解析，GUI shell 脚本语法及 diff 空白检查通过。

以下为核对已有本地报告得到的历史验收结果，本次版本整理未重新启动 Isaac 验收或 PPO 训练：

- P1 五组独立检查全部通过：862 个动作、159,480 个起始姿态候选，代表动作 210 次真实 reset，以及 998 个控制步的课程/续训审计。
- 高度扫描验证通过：280 次出生放置、350 次独立 PhysX 查询；最大 PhysX 高度差约 4.29 μm。
- GUI 训练 `acceptance_gui_20261010_r14` 保留 49 次更新的启动验收快照；最终状态为 **`stopped_by_user`**，2026-10-10 19:43:51 停止，共 **390 次 PPO 更新、9,360 个控制步、3,572 次 reset/sampler outcome**。
- 最后一个常规保存文件为该运行目录下的 `model_15350.pt`；状态文件的最后迭代为 15389，未把尚未落盘的最后进度当作 checkpoint。

### 复现条件与范围

查看 [使用命令](docs/USER_READ.md)、[P1 验收记录](docs/PGMT_P1_ACCEPTANCE.zh-CN.md)、[高度扫描](docs/PGMT_HEIGHT_SCAN.zh-CN.md)、[地形课程与续训](docs/PGMT_TERRAIN_CURRICULUM.zh-CN.md) 和 [Stage 2 计划](docs/PGMT_STAGE2_PLAN.zh-CN.md)。

本版本是 **P1 固定第 0 帧初始化与无感知地形 PPO**。扫描接口已具备，但无感知训练入口关闭扫描器，actor/critic 未使用地形输入。Terrain-Glimpse/IFM、第四个价值头、地形接触奖励、恢复机制、五类语义组等比例分配及完整 Stage 2 性能验证仍待完成。初始化检查与训练可运行不等于策略已在全部困难地形收敛。

地形 `pgmt_stage2_v5`、带标签动作库 `pgmt_clipped_50hz_tcrs_v1`、G1 资产、Stage 1 权重和 GUI 使用的本地 G1 USD 仍需按对应文档准备。生成的网格、NPZ、训练权重和原始日志未上传；版本快照保存其关键哈希与结果摘要。GUI 封装的默认环境路径适用于当前机器，可通过脚本中的 `PGMT_*` 变量调整。

获取标签对应代码并创建独立工作分支：

```bash
git fetch origin --tags
git switch -c work/pgmt-p1-20261010 pgmt-p1-2026.10.10
```

## 2026-10-10 — STAGE 2 地形与初始化基线

提交 [`7544eab`](https://github.com/bala-ba/PMT/commit/7544eaba24dc0b894ec0512a6490a0c25dc39aff)：七种样式、十级地形的生成/导入/可视化，统一参考坐标变换与机器人位置初始化，碰撞及边界检查，离线 TCRS 接触标签和配套文档；当时 25 项相关测试通过。

## 2026-10-08 — PGMT Stage 1 基线

提交 [`f8aff56`](https://github.com/bala-ba/PMT/commit/f8aff5691bb63b34c68f886de7691ba6645ffbf3)：PGMT Stage 1、深度感知及 MuJoCo 评估工具。后续地形初始化与 P1 训练基于此阶段的网络和已有权重演进。
