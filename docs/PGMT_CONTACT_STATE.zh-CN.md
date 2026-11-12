# PGMT P3 接触状态与事件接口

日期：2026-10-14；状态更新：2026-10-15。训练期基础状态、计时和事件已供 T1–T6 读取；正式 `pgmt_stage2_terrain` 已接入 27 项奖励和四头，原 `pgmt_stage2` 保留 21 项/三头。Stage2 默认启用本接口，Stage1 和 P1 独立任务保持原行为。以下专项证据保留接口逐步接入时的验证范围。

T1 已在 2026-10-18 改为消费本接口的实际 `support/support_valid` 和同帧参考标签：双脚整体匹配时持续结算支撑质量。`touchdown` 事件保留供调试，不再触发 T1 奖金；正式四头 PPO 的 terrain 组及原奖励接口保持。见 [T1 接口与结果](PGMT_TOUCHDOWN_QUALITY.zh-CN.md) 和 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。

T2 reference contact match 也已实现并验收：用户确认使用本接口的 `contact/contact_valid`（全对象力范数），与冻结的同帧参考标签持续匹配。见 [T2 接口与结果](PGMT_REFERENCE_CONTACT_MATCH.zh-CN.md)。

T3 slip 已实现并验收：使用 `contact/contact_valid` 和冻结的 `foot_body_lin_vel`，计算实际接触脚的世界 XY link 原点速度平方和。见 [T3 接口与结果](PGMT_SLIP.zh-CN.md)。

T4 stumble 使用冻结的 `net_forces_w/contact_valid`，持续判断 `||Fxy|| > 5|Fz|+10 N`，不乘参考标签或承重状态。计算、缓存和独立验收见 [T4 接口与结果](PGMT_STUMBLE.zh-CN.md)。

T5 contact switching 复用 `fast_contact_switch/contact_history_valid`，只在旧状态年龄小于 0.1 s 且已过前 5 步保护的双向切换当步计罚，不新增或重复推进计时器。计算、缓存和独立历史重放验收见 [T5 接口与结果](PGMT_CONTACT_SWITCHING.zh-CN.md)。

T6 contact force 使用冻结的 `net_forces_w/contact_valid`，按 XYZ 三分量平方逐脚求和，无力阈值或接触状态门控。计算、缓存和六项联合验收见 [T6 接口与结果](PGMT_CONTACT_FORCE.zh-CN.md)。

## 1. 力来源与状态定义

- `net_forces_w`：原生全身接触传感器中双脚的法向接触力合力，含所有碰撞对象；不包含摩擦力。
- `terrain_forces_w`：左右脚各自一个原生 ContactSensor，只过滤有效训练地形碰撞体，对 `force_matrix_w` 的对象维求和。覆盖所有 tile，排除机器人、其他对象与外边界地形；不绑定出生 tile。
- `contact = ||net_forces_w|| > 10 N`：T2/T3 使用此实际接触状态，T5 使用它的快速切换事件；T4 直接比较同一全对象力的水平/竖直分量；T6 对该力无阈值地求平方和。
- `support = terrain_forces_w[...,2] > 10 N`：已确定的承重定义，使用世界 Z 正分量，供 touchdown 使用；不取范数、绝对值或脚局部 Z。

当前不增加 10/5 N 滞回、足底近地条件、脚速门槛或摩擦力通道。原生传感器的 air-time 状态仍基于其力范数阈值，不能拿来替代本接口的承重计时。

配置位于 [stage2.py](../pmt_tasks/env_cfgs/pgmt/stage2.py)，状态机位于 [pgmt_contact_state.py](../pmt_tasks/pgmt_contact_state.py)。阈值集中在 `CONTACT_STATE_CONFIG`，通过 `cfg.pgmt_contact_events` 保存到环境配置；非法配置启动即报错。

| 配置 | 当前值 |
| --- | ---: |
| `contact_threshold_n` | 10 N，严格 `>` |
| `support_threshold_n` | 10 N，严格 `>` |
| `touchdown_min_non_support_steps` | 2 个控制步 |
| `fast_switch_seconds` | 0.1 s，严格 `<` |
| `switch_grace_steps` | 前 5 个控制步不产生快速切换事件 |

地形过滤使用 manifest 生成的**显式碰撞体路径列表**，在运行时与实际导入 USD 的碰撞体列表逐项核对。v6 为 800 个，v5 为 70 个。共享地图的通配路径会被 PhysX 按环境配对解析，不适合作为一个全局过滤表达式；不能改回全局通配或未过滤净力作为替代。

## 2. 时间、事件与无效状态

左右脚独立，控制步长读取 `env.step_dt`，当前为 0.02 s。计时使用整数控制步累计，再换算成秒；不按 0.005 s 物理子步推断额外事件。

| 字段 | 含义 |
| --- | --- |
| `contact/support`、`previous_contact/previous_support` | 当前状态和有效历史状态 |
| `contact_valid/support_valid` | 对应力通道是否有限、当前样本是否有效 |
| `contact_history_valid/support_history_valid` | 当前及上一份数据均有效且属于同一回合，即各通道的 b |
| `contact_started/contact_ended/contact_switched` | 原始接触的上升沿、下降沿、任意切换 |
| `support_started/support_ended/support_switched` | 承重状态的上升沿、下降沿、任意切换 |
| `touchdown` | 承重上升沿，且此前连续未承重至少 2 步；不代表高度质量已经合格 |
| `fast_contact_switch` | 接触切换，且更新前接触状态年龄 <0.1 s、本回合控制步数 >5 |
| `contact_age/support_age` | 自首次有效观测/最近状态切换后的年龄；首次及切换当步为 0 |
| `previous_contact_age/previous_support_age` | 本次更新前的年龄，用于判断刚结束的状态 |
| `contact_time/air_time` | 当前连续接触/非接触观测步数乘 dt，不对应的一个为 0 |
| `support_time/non_support_time` | 当前连续承重/未承重观测步数乘 dt，不对应的一个为 0 |
| `last_contact_time/last_air_time` | 最近结束的接触/非接触段已观测时长 |
| `last_support_time/last_non_support_time` | 最近结束的承重/未承重段已观测时长 |
| `previous_air_steps/previous_non_support_steps` | 更新前的连续非接触/未承重步数 |

以上状态、计时、事件字段形状均为 `[num_envs,2]`，脚顺序固定左、右。力与足位置/速度为 `[num_envs,2,3]`，足姿态为 `[num_envs,2,4]`、wxyz。

年龄与观测段时长有意区分：首次观测到支撑时 `support_age=0`、`support_time=dt`；持续第二步时分别为 `dt`、`2dt`。前者复现文档中事件年龄口径，后者用于报告已累计的观测步数。reset 时中途开始观测到的接触段不等于完整物理接触持续时间。

未承重不等于已经离地：轻触、侧撞或向下受力也可能 `support=False`。力出现 NaN/Inf 时，只使对应脚/对应通道无效、清除该通道历史，不产生开始/结束/切换事件；再次获得有限力时重新建立基线。`False` 与 `valid=False` 表示未知，不能用于推断摆动。缺少过滤传感器、错误 body 顺序、零过滤列或错误形状直接报错，不回退到其他力来源。

## 3. 更新和 reset 顺序

[PGMTState.rewards()](../pmt_tasks/pgmt_runtime.py) 在原生 RewardManager 第一次计算本控制步奖励时，读取物理状态、当前参考和两种力，更新一次接触状态。原有 21 项 reward 的公式、权重和 dt 均不改变。

```text
物理步完成 → 生成接触/承重事件 → 保存本步状态与参考快照
          → 计算旧回合奖励 → done/reset → command 推进/新观测
```

同一步重复读取奖励或接口不会推进计时、重采样或重复生成事件。状态机拒绝倒序或跳过控制步的更新，防止把未观测时间误计入接触时长。

reset 仅清理指定环境的当前历史、计时、事件和参考快照，标为无效；同一步再次读取不能用旧物理力重新初始化。下一控制步的首份测量只建立基线，无 touchdown/switch。其他环境的历史保持不变。

终止步已结算的事件与参考另存为 reward snapshot，reset 不覆盖它。此快照携带本步采样的 `reference_motion_ids/reference_frame_ids` 和参考标签，避免读取 command 推进或 reset 后的新参考。事件不因跨 tile 自动清零；力源覆盖整张有效地图，出生课程等级仍由原课程接口管理。

## 4. 后续奖励的读取方式

```python
from pmt_tasks.pgmt_runtime import get_pgmt_state

state = get_pgmt_state(env)
state.rewards()  # 原生 PGMT reward 已在此结算时，无需重复调用；重复调用也不会推进
sample = state.contact_state(for_reward=True)

support = sample["support"]
touchdown = sample["touchdown"]
switch = sample["fast_contact_switch"]
reference_valid = sample["reference_contact_valid"]
reference_contact = sample["reference_contacts"]
foot_pos_w = sample["foot_body_pos"]
```

- `contact_state()`：当前可读快照；reset 行已无效，适合调试。读取本身不采样传感器。
- `contact_state(for_reward=True)`：本控制步冻结的奖励快照；允许在同一步 reset 后读取旧回合数据。尚未结算本步时明确报错，不返回上一控制步冒充当前奖励数据。
- 两种调用均支持 `env_ids` 一维索引/slice，返回独立张量副本，调用方不能改坏缓存。
- `sample_step/episode_step/episode_id` 形状为 `[num_envs]`。reset 后当前 `sample_step=-1、episode_step=0`，`episode_id` 增加；下一步从 1 计数。
- `robot_foot_terrain_forces()` 是即时力读取接口，不负责推进状态。奖励优先使用已冻结的 `terrain_forces_w`。

这些计时和事件属于物理回合瞬时状态。续训会重置物理回合，因此重新初始化，不从 checkpoint 恢复中断时的脚部接触历史。

## 5. 验证记录

单元/回归入口：`tests/test_pgmt_contact_state.py`、`tests/test_pgmt_contact_reference.py`、`tests/test_pgmt_mdp.py`。覆盖力方向/过滤通道、严格阈值、首次测量、两步未承重、年龄边界、初始宽限、NaN/Inf、部分 reset、快照防修改、参考时序、重复 RewardManager 读取及 CPU/CUDA 一致性。

Isaac 入口：

```bash
bash scripts/validate_pgmt_contact_state.sh \
  --log_dir logs/pgmt_p3_contact_state/<新的验收目录>
```

2026-10-14 最终验收结果：

- **70 passed**：使用 IsaacSim Python、禁用 pytest 第三方自动插件，运行上述三组测试及 `test_pgmt_config.py`、`test_pgmt_curriculum.py`、`test_pgmt_terrain_query.py`、`test_pgmt_reference.py`。包括真实 CUDA 上的状态机与 CPU 对照、旧 21 项奖励回归、帧/课程/地图接口回归。
- **Isaac passed**：[最终报告](../logs/pgmt_p3_contact_state/isaac_03/training_status.json)。v6 五类地形各 1 环境、L0、headless，共 67 个控制步，完成后关闭仿真。每只脚的过滤矩阵为 `[5,1,800,3]`；两个传感器均核对了全部 800 个实际碰撞体。五类地形均观测到正向承重力，前 40 步共产生 10 次 touchdown、10 次接触开始、1 次接触结束和 1 次快速切换。
- **非地形对象隔离通过**：额外高台上的双脚全接触力最大约 820.64 N，而地形过滤力始终为 0 N，无承重或 touchdown 事件。
- **时序通过**：受控终止步保留旧回合参考和事件；局部 reset 不影响其他环境，新回合首次测量无伪事件；同一步重复读取不增加计数。

首次 Isaac 验证发现全局过滤通配符按环境数量配对，报告 `expected 5, found 50`。已改为 manifest 生成的显式碰撞路径，并与导入 USD 列表及 PhysX `filter_count` 核对；后续两次有限步验证通过。没有升级环境依赖或以未过滤力绕过问题。

本次验证的是基础状态和事件接口，不包含 PPO 训练、六项新奖励数值或学习性能。真实物理验证为 5 环境 L0；v5 检查了 70 条过滤路径生成，未重跑 v5 仿真，也未验证 8,192 环境下新增传感器的吞吐/显存。

## 6. 平地交互检查

```bash
bash scripts/view_pgmt_contacts.sh \
  --log_dir logs/pgmt_p3_contact_viewer/$(date +%Y%m%d_%H%M%S) \
  --speed 0.5
```

默认使用 `walk1_subject1_clip0.npz`、严格加载 `model_15000.pt`。可通过 `--motion /绝对路径/动作.npz` 切换带离线标签的动作。真实机器人由 Stage1 策略驱动，在 v6 flat/L0 上仿真；接触传感器、状态机和奖励读取时序复用 Stage2 实现。仅该检查入口固定平地出生 XY/yaw、停用地形课程和随机扰动/观测噪声；没有修改训练配置或更新模型。保留原失败/片段耗尽等终止逻辑。

- **真实机器人脚部球**：绿＝地形向上力承重，灰＝未承重，品红＝测量尚无效。
- **旁侧参考骨架脚部球**：青＝离线支撑，蓝＝离线摆动，品红＝离线标签无效。骨架仅显示，沿世界 Y 平移 1.2 m，不参与碰撞；这段平移不施加到真实 reference 或机器人。
- **黄色球**：最近 touchdown 的 0.30 s 显示高亮，方便肉眼观察；不延长底层事件。面板的 `Touchdown NOW` 和 CSV 才是严格的单控制步事件。
- 面板分别显示左/右脚的接触、承重、向上力、承重/未承重时长、事件、离线标签和最近事件的回合/帧号。`Pause / Run`、`Step 20 ms`、`0.25x / 0.5x / 1x` 用于暂停、单步和慢放，物理/控制 dt 均不变；`Reset frame 0` 重新初始化，`Follow / Free` 切换相机跟随。

显示的离线标签来自 `contact_state(for_reward=True)` 冻结的参考帧，参考骨架在 `env.step()` 前读取同一帧，并断言帧号一致。遇到自动 reset，终止步结果保留在 CSV/事件列表，视口切换为新回合的未知承重状态及第 0 帧参考，避免把旧力标到新位姿上。CSV 每控制步每脚一行，记录力、有效性、状态、计时、事件、离线标签、帧号和 done；面板暂停时不产生额外控制样本。

日志目录包含 `contacts.csv`、`training_status.json`、实际 `env.yaml` 和 `viewport.png`。离线标签描述参考动作，真实力描述策略实际运动，两者不一致需要结合姿态跟踪误差和事件时序判断，不能直接认定检测实现错误。

2026-10-14 GUI 入口复验：`check_01` 完成 550 个控制步，覆盖完整 499 帧步行动作及一次自动 reset；保存 1,100 条单脚记录，落脚事件左 5 / 右 4 次。逐行复核力阈值、两步未承重门槛、快速切换年龄/宽限与 reset 基线通过；终止步参考为 498，下一回合从 0 开始，见 [检查记录](../logs/pgmt_p3_contact_viewer/check_01/viewer_check.json)。已检查实际视口截图中的机器人、参考骨架和脚部颜色标记。
