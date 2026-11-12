# PGMT 接触标签：离线 TCRS 与训练期接口

日期：2026-10-09；训练期接口更新：2026-10-10。状态：离线生成器与训练期标签接口已实现，当前 CSV 目录已生成 **862 个 segment、430,138 帧（50 Hz）**；terrain-contact 奖励公式待 P3 接入。

## 判定依据

TCRS 的 `MotionClip` 实际使用脚踝高度、脚踝世界速度和连续段去抖识别支撑/摆动：

1. 左右脚共用高度阈值 `max(0.06 m, P25(两脚高度) + 0.02 m)`。
2. 候选支撑条件为 `height < threshold` 且 `speed < 0.5 m/s`。
3. 删除少于 **4 帧**的支撑段；填补两侧均为支撑、少于 **2 帧**的非支撑间隙。片段首尾不循环拼接。
4. `gait_phase.py` 的周期相位时钟只作辅助，原 `MotionClip` 明确不让它覆盖接触 mask，离线标签沿用这一约定。

这段逻辑已提取至 [contact_detection.py](../TCRS/stair_mppi/contact_detection.py)，[原生 MotionClip](../TCRS/stair_mppi/minimal_mppi_demo.py) 和离线生成器共同调用。默认行为与原实现一致；500 组随机序列的左右脚结果逐帧相同。在 50 Hz 下，4 帧对应 80 ms、2 帧对应 40 ms；阈值和连续段长度均可配置。

## 网格查询与有效性

[pgmt_contacts.py](../pmt_tasks/pgmt_contacts.py) 对**参考动作所在表面**进行真实三角网格向下射线查询，用 `ankle_z - mesh_height` 替代绝对脚踝高度，因此允许负高度表面。每只脚同时查询 G1 XML 中四个碰撞球的底部；球心随脚旋转，球底按世界竖直方向减半径。当前球心来自 TCRS G1 的脚跟/脚尖四角，半径为 5 mm，未使用硬编码的“脚踝离地 6 cm”替代碰撞几何。

在 TCRS 判定基础上增加以下网格约束，属于本项目的明确工程设定：

- 至少一个足底探针距离网格不超过 **2 cm**，才有支撑候选；仅凭低速度和相对百分位不能把静止悬空的脚判为支撑。
- 任一足底探针穿透超过 **2 cm**，或脚踝/任一足底射线缺失，则该脚该帧为 **unknown**，由 `contact_valid=False` 表示。
- 先对候选应用 TCRS 去抖，再约束最终标签仍满足网格条件，避免短间隙填补产生悬空接触。
- 保存网格约束前的 `contacts_tcrs` 供对照；不删除 clip，不改写关节轨迹，不把无效帧当作摆动脚负样本。

CSV 的参考表面首版设为 **z=0 平面三角网格**，覆盖全部输入轨迹。网格保存为 `reference_mesh.ply` 并记录哈希。可通过 `--mesh` 查询其他已经与动作世界坐标对齐的参考网格；不把同一平地动作直接批量放到七种训练地形上生成配对标签，也不调用 TCRS 的地形变形、落脚点规划或 IK。

论文公开了离线 terrain-mesh 接触查询，但未给出具体阈值协议；上述 TCRS 判定与网格约束是可审计的复现选择。这里的“支撑”是运动学/几何标签，不是力传感器实测承重。

## 数据与文件约定

输入：`/home/lzp/projects/perceptive_tracking/data/lafan_prepared/clipped_csv`。生成前核验 CSV 的 manifest、文件哈希、29 个关节名称、xyzw 四元数和实际时间轴。

流程：**30 Hz CSV → 50 Hz 插值 → PMT G1 FK/速度 → 网格查询 → TCRS 标签 → NPZ**。根平移和关节角使用线性插值，根旋转使用 SLERP；转换为 PMT 的 BFS 关节/刚体顺序和 wxyz 四元数。接触标签在重采样后重新计算，不对二值 mask 做线性插值。世界角速度由相邻刚体旋转差计算。

300 帧源数据覆盖 `0..9.9666667 s`，因此每个输出为 **499 帧，时间轴 `0..9.96 s`**。不外推第 500 帧，不拼接不连续片段。后续 command 必须处理片段结束和未来参考不足，不能以重复末帧伪造连续动作。

输出：`assets/motions/pgmt_clipped_50hz_tcrs_v1/`，每个 CSV 对应一个 NPZ，同时包含 PMT 原生 motion 数组和下列标签：

| 字段 | 形状/含义 |
| --- | --- |
| `contacts` | `[N,2] float32`，左、右脚；1 支撑，0 非支撑，必须结合有效性使用 |
| `contact_valid` | `[N,2] bool`，训练接触匹配项必须屏蔽 False |
| `swing` | `[N,2] bool`，仅在有效帧上等于 `~contacts` |
| `foot_state` | `[N,2] int8`，−1 unknown、0 摆动、1 支撑 |
| `support_mode` | `[N] int8`，−1 unknown、0 两脚均无接触、1 左、2 右、3 双脚；0 不代表身体其他部位也未接触地面 |
| `contacts_tcrs` / `contacts_raw` / `mesh_support` | 网格约束前的 TCRS 去抖结果、约束后去抖前候选、几何支撑候选 |
| `foot_ground_height` / `foot_probe_clearance` / `foot_mesh_hit` | 网格高度、四个球底的有符号竖直间隙、射线有效性 |
| `time` / `source_frame` / `segment_id` | 输出时间、源序列浮点帧坐标、`motion_name:start:end_exclusive` 稳定标识 |
| `foot_names` | 固定 `[left_ankle_roll_link, right_ankle_roll_link]` |

`manifest.json` 记录每个 clip 的哈希、源帧区间、接触/无效计数、抽样 FK 核验、机器人/算法/网格指纹和参数；它在全部输出成功后写入。`MotionDataStore` 将附加数组保存在 metadata 中，PGMT command 据此构建下述训练期接口；保持一个 CSV 一个采样 segment，`chunk_length=0`。

## 训练期接口（2026-10-10）

[ContactLabelStore](../pmt_tasks/pgmt_contact_reference.py) 按 command 的实际 motion 顺序打包 `contacts` 和 `contact_valid`，启动时校验字段、帧数、二值范围和足部名称，并将足部顺序统一为 **左、右**。标签一次性放到仿真设备，每步只进行张量索引，不读取 NPZ；关闭 reset 的足部支撑过滤不会关闭标签加载或清除无效 mask。

在后续 reward 函数内读取 [PGMTState](../pmt_tasks/pgmt_runtime.py)：

```python
from pmt_tasks.pgmt_runtime import get_pgmt_state

state = get_pgmt_state(env)
current = state.reference_contacts()               # 当前帧；contacts: [N, 2]
future = state.reference_contacts(future=True)      # 未来窗口；contacts: [N, 8, 2]
valid = current["contact_valid"]                    # 必须用于屏蔽无效监督
stance = current["stance"]                         # 有效且期望接触
swing = current["swing"]                           # 有效且期望不接触
forces_w = state.robot_foot_contact_forces()        # 实际净接触力 [N, 2, 3]，世界系，单位 N
```

两个接口均支持 `env_ids` 一维索引列表/张量或 slice，用于局部环境查询。`state.foot_names` 为 `(left_ankle_roll_link, right_ankle_roll_link)`；`foot_body_indices` 用于机器人/全身参考数组，`foot_sensor_indices` 用于 contact sensor。仿真实际接触力按同一左右顺序返回，力读取接口本身不施加二值阈值。P3 已新增 [实际接触状态与事件接口](PGMT_CONTACT_STATE.zh-CN.md)，集中维护接触/承重判据、计时和事件；后续奖励通过 `contact_state(for_reward=True)` 读取本步冻结的实际状态与参考标签，避免 command 推进或 reset 后标签错位。

标签字典的字段如下；令 `S=[N]`（当前）或 `S=[N,8]`（未来），指定 `env_ids` 时 N 为选中环境数：

| 字段 | 类型与形状 | 语义 |
| --- | --- | --- |
| `contacts` | float32，`S+[2]` | 原始 0/1 目标，单独读取不能区分无效标签 |
| `contact_valid` | bool，`S+[2]` | 源标签有效且请求帧处于片段范围内 |
| `stance` / `swing` | bool，`S+[2]` | `valid & contact` / `valid & ~contact`，无效时两者均 False |
| `foot_state` | int8，`S+[2]` | −1 无效、0 非接触、1 接触 |
| `support_mode` | int8，`S` | −1 任一脚无效；否则 0 两脚无接触、1 左、2 右、3 双脚 |
| `frame_valid` | bool，`S` | 请求索引是否落在真实片段内，与单脚源标签有效性独立 |
| `motion_ids` | int64，`S` | command 实际 loader 中的 motion id |
| `frame_ids` / `requested_frame_ids` | int64，`S` | 边界截取后的读取帧 / 未截取的请求帧 |

当前/未来位姿和接触标签共用 command 的 motion/frame 索引及 `[0,1,3,7,15,31,63,127]` 偏移。超过片段末尾时，读取索引截到末帧，但 `frame_valid=False`、`contact_valid=False`；不能把重复末帧用于有效接触监督。每次查询返回独立标签快照，因此同一步中局部 reset、换片后会立即读取新索引，先前保存的终止帧标签不会变化。

**奖励使用约定：** 在原生 RewardManager 的计算阶段读取接口，此时参考仍对应本步，之后才执行 done/reset 和 command 推进；不要在 `env.step()` 返回后读取标签来重算上一时刻奖励。接触匹配项必须按 `contact_valid` 屏蔽并仅以有效脚数归一化，全无效时该项贡献为 0。当前 `stance/swing` 已屏蔽无效位，但仍不能用 `~stance` 代替 `swing`。

这些是离线参考动作的接触意图，平移/yaw 放置不会改变标签；实际训练地形上的接触由传感器反馈。爬行时两脚无接触不代表膝/手无支撑，当前接口不推断膝/手标签，不修改原始动作或重定向结果。PGMT command 在标签缺失或错位时启动报错；stage 1 仍使用原有 command，显式调用本接口才会提示缺少带标签数据。

运行期验证入口（同时回归动作级采样及固定第 0 帧 reset）：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_contact_reference.py tests/test_pgmt_sampling.py \
  tests/test_pgmt_reference.py tests/test_pgmt_mdp.py tests/test_pgmt_config.py

/home/lzp/miniconda3/envs/pmt/bin/python scripts/validate_pgmt_sampling.py \
  --headless --device cuda:0 --output logs/pgmt_contact_interface
```

单元/回归测试共 38 项通过，覆盖左右脚重排、变长片段末尾 mask、无效标签、局部 reset、同一步换片和独立快照。Isaac [验证报告](../logs/pgmt_contact_interface/validation.json) 已通过：862 段动作共 430,138 帧的标签与源数据逐帧一致，2,623 个无效单脚×帧样本保留 mask；七种地形下执行 11 个物理步、28 次环境 reset 采样，并验证片段末帧的 reward 读取先于 reset/command 推进、未来越界 mask 和三个爬行片段的标签读取。该脚本的零值 reward 探针只用于验证，没有添加生产环境接触奖励；这些短程检查不代表全库全地形的物理放置验收。

## 生成与检查

在 PMT 项目根目录运行：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python scripts/prepare_pgmt_contact_labels.py
```

默认输出已存在时会拒绝隐式覆盖；重建同一版本可显式加 `--overwrite`。

从 `pgmt-p1-2026.10.10` 起，输出目录中的人工 `motion_terrain_compatibility.csv` 随 Git 保存。因此全新 checkout 也会包含这个非空目录；首次生成默认 v1 数据需显式使用 `--overwrite`，或先生成到一个新目录。该选项会重建派生 NPZ、参考网格和 manifest，不修改兼容 CSV；使用前确认目录中的派生数据确实需要重建。原始动作 CSV 及其 conversion manifest 仍需在本地准备。

人工筛选后可以提供 JSON 文件名列表，例如 `["walk1_subject1_clip0.csv", "jumps1_subject1_clip0.csv"]`：

```bash
/home/lzp/miniconda3/envs/pmt/bin/python scripts/prepare_pgmt_contact_labels.py \
  --selection selected_segments.json \
  --output assets/motions/pgmt_clipped_50hz_tcrs_v2
```

```bash
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt_contacts.py tests/test_pgmt_data.py
```

已通过 19 项测试，覆盖 TCRS 去抖、左右脚/双支撑、静止悬空、负高度网格、缺失射线、穿透、旋转足底几何、30→50 Hz 时间边界、四元数符号和世界角速度、FK、保存字段及哈希。现有 stage 1 数据转换测试同时通过。

本次全量资产检查结果见 [validation.json](../logs/pgmt_contacts_v1/validation.json)：

- 862 个文件的输入/输出哈希、代码/机器人/网格哈希及逐帧数据约定全部通过，原始 CSV 保持不变，缺失射线为 0。
- 左/右脚支撑帧数分别为 **210,730 / 211,475**。
- **2,623 个单脚×帧样本（0.305%）**因穿透超过 2 cm 标为 unknown，分布于 135 个片段；其余数据继续保留。无效性是标签有效性标记，不是自动动作筛选。
- walk、run、sprint、jumps、dance、fight、fightAndSports、fallAndGetUp 各抽一个真实片段，网格约束前的结果与原生 `MotionClip` 完全一致。
- 对上述片段共 136 个状态，直接读取 MuJoCo 碰撞球世界坐标独立核验足底间隙，最大误差 **5.86×10⁻⁸ m**。

## 仿真器回放

使用同一 G1 模型在 MuJoCo 中回放保存的参考姿态；脚部颜色直接读取 NPZ 标签：绿色支撑、蓝色摆动、紫色无效。回放前逐帧核验刚体 FK，属于参考轨迹可视化，不是策略控制结果。

```bash
MUJOCO_GL=glfw /home/lzp/miniconda3/envs/pmt/bin/python scripts/view_pgmt_motion.py \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/dance1_subject1_clip0.npz
```

窗口循环播放；空格暂停/继续，左右方向键逐帧，R 回到开头。导出单次完整视频：

```bash
MUJOCO_GL=egl /home/lzp/miniconda3/envs/pmt/bin/python scripts/view_pgmt_motion.py --video
```

默认输出位于 `logs/pgmt_motion_preview/dance1_subject1_clip0/`。本次视频包含全部 499 帧、50 fps，编码时长 9.98 s；全帧 FK 最大误差为 2.17×10⁻⁷ m。
