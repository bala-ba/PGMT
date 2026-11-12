# PGMT P3：奖励审计与实施建议

日期：2026-10-13；奖励讨论及公式一致性更新：2026-10-14。依据：项目内 `PGMT_LITERATURE.pdf` 第 4 页表 I / 恢复课程、第 5 页地形容差、四头 critic 和课程说明；当前 PMT 源码、P2 正式任务配置和实际使用的机器人 USD。已渲染核对论文第 4–5 页。

**当前总验收结论（2026-10-14）：P3 部分通过，尚未全部完成。** `pgmt_stage2_terrain` 的 27 项奖励、三项下肢容差和四头已接入；P3.1–P3.4 实现与有限链路通过。P3.5 头/胸真实碰撞链路已补验通过，但发现短冲击峰值采样限制；爬行接触冲突统计及 P3.6 分地形/难度策略性能仍待完成。最新双腕 EE 奖励、0.2 s 终止容忍和 10 s 回合配置下，600 项 CPU/CUDA 回归、160 环境 3+2 次 PPO 更新、独立进程二次恢复及 Stage1/P2 迁移通过，详见 [P3 总验收](PGMT_P3_ACCEPTANCE.zh-CN.md)。原 P2 保留 21 项/三头；首轮旧配置记录保留于 [P3.4 验收](PGMT_P34_FOUR_HEAD_ACCEPTANCE.zh-CN.md)。论文未公开的本地阈值不称为论文参数。

**本轮范围更新（用户决定，2026-10-14）：暂不构建或接入 recovery pool，recovery upward velocity 及配套恢复课程暂缓。** P3 本轮目标为 27 项奖励（6 upper + 6 lower + 6 terrain + 9 auxiliary）及四头；recovery 不阻塞本轮完成。其公式保留为论文差异/后续参考，不计为已实现，也不自动安排到本轮训练或验收。

原审计阶段只研究、核对和更新文档。2026-10-14 已在 Stage2 接入共享接触状态、地形过滤力、计时与事件，并完成 CPU/CUDA 和有限步 Isaac 验证，详见 [接触状态接口与验收](PGMT_CONTACT_STATE.zh-CN.md)。基础状态阶段未启用新 PPO 奖励或价值头，未启动 PPO 训练。

同日后续：**T1–T6 六项 terrain-contact 的数值核、运行期缓存和正式四头 RewardManager 接入均已完成**，详见 [T1 实现与验收](PGMT_TOUCHDOWN_QUALITY.zh-CN.md)、[T2 实现与验收](PGMT_REFERENCE_CONTACT_MATCH.zh-CN.md)、[T3 实现与验收](PGMT_SLIP.zh-CN.md)、[T4 实现与验收](PGMT_STUMBLE.zh-CN.md)、[T5 实现与验收](PGMT_CONTACT_SWITCHING.zh-CN.md)、[T6 实现与验收](PGMT_CONTACT_FORCE.zh-CN.md)。独立验收与新正式任务均为 27 项/四组；原三头任务仅保留接触诊断率，不将新项混入三头总奖励。

**完整公式索引：见本文第 7 节。** 逐项列出 12 项身体/关节跟踪、10 项辅助项（含已暂缓的 recovery）及 6 项 terrain-contact，共 28 项供对照；本轮排除 A4，目标为 27 项。各项标明现有实现、P3 候选或暂缓状态。前三节保留审计结论；第 7 节用于后续实现时核对符号、归约方式、权重、掩码和 dt。

## 1. 当前真正生效的计算路径

下图为保留的 P2 三头基线。正式 `pgmt_stage2_terrain` 选择 `pgmt_stage2_terrain` 奖励配置，使用 `PGMTTerrainTrainingEnvCfg`、27 项 RewardManager term 和四组贡献；新增 T1–T6 在同一控制步缓存中计算，单次乘权重/dt 与旧回合结算原则相同。

```text
configs/task/pgmt_stage2.yaml: reward = pgmt_stage1
  → builder._build_pgmt_stage2_env() 注入权重、tracking 参数
  → PGMTPerceptionEnvCfg → PGMTPlacementEnvCfg → PGMTStage1EnvCfg
  → PGMTRewardsCfg：21 个独立 RewardManager term
  → mdp.pgmt_rewards.pgmt_reward()
  → PGMTState.rewards()：一个控制步只计算一次，更新前一帧缓存
  → pgmt_math.reward_terms()：返回未加权、未积分的奖励率
  → RewardManager.compute(dt)：weight × rate × dt，形成标量 reward
  → wrapper 从冻结的 _step_reward 取三组贡献，写 extras["reward_groups"]
  → PPO：三组各自计算 GAE/return，actor 优势先求和再归一化
```

代码入口：[任务配置](../configs/task/pgmt_stage2.yaml#L9)、[builder](../pmt_tasks/builder.py#L376)、[奖励配置类](../pmt_tasks/env_cfgs/pgmt/stage1.py#L50)、[运行时缓存](../pmt_tasks/pgmt_runtime.py#L216)、[核心公式](../pmt_tasks/pgmt_math.py#L140)、[奖励分组](../pmt_tasks/pgmt_runtime.py#L243)、[wrapper](../pmt_tasks/isaaclab_rl/rsl_rl/vecenv_wrapper.py#L153)、[GAE](../motion_tracking_rl/storage/rollout_storage.py#L143)。

当前控制步长为 `4 × 0.005 = 0.02 s`。`_step_reward` 保存加权率；wrapper 对其乘 dt 是重建已经结算的分组贡献，并非再次修改标量奖励。P3 保留“只积分一次”的约定。终止步先计算旧回合奖励，随后 reset；不得用 reset 后的参考/接触重算旧奖励。

### 现有 21 项

跟踪公共核：`E(e,σ) = exp(-mean(e_j²)/σ²)`。每个 link 的位置/速度误差先求三维范数；姿态误差为四元数夹角；关节误差为逐关节差。身体位置减去各自 torso anchor，并在各自 anchor 的 yaw 坐标系表达；身体姿态和速度也按对应 yaw 对齐。关节使用原始参考轨迹。

| 项目 | 上肢权重 | 下肢权重 | 当前尺度 σ | P3 处理 |
| --- | ---: | ---: | ---: | --- |
| Link position | 1.0 | 0.5 | 0.15 m | 下肢增加地形容差 |
| Link orientation | 1.0 | 2.0 | 0.4 rad | 下肢增加地形容差 |
| Link linear velocity | 0.5 | 0.5 | 1.0 m/s | 保留 |
| Link angular velocity | 0.5 | 0.5 | 3.14 rad/s | 保留 |
| Joint position | 1.0 | 0.5 | 0.4 rad | 下肢增加地形容差 |
| Joint velocity | 0.5 | 0.5 | 5.0 rad/s | 保留 |

集合由 [body_groups()](../pmt_tasks/pgmt_math.py#L122) 按名称解析：上肢 link 为 torso/shoulder/elbow/wrist，下肢为 hip/knee/ankle；上肢关节含 waist，下肢为腿部关节。与 command 用于其他管理逻辑的 14 个 tracked bodies 区分，奖励读取完整 30 刚体参考。

| Auxiliary 项 | 权重 | 当前计算 |
| --- | ---: | --- |
| Root orientation | +0.5 | torso 世界姿态夹角的指数奖励，σ=0.4 rad |
| Corrected root velocity | +2 | 参考 anchor 坐标系下 XY 速度误差，σ=1 m/s；目标是修正后速度 |
| Floating-anchor position | +1 | 世界 XY anchor 距离，0.3 m 死区，σ=0.5 m |
| Pelvis vertical acceleration | −0.001 | pelvis 竖直速度差分的平方，上限 10000 |
| EE acceleration mismatch | −0.001 | Stage2 仅双腕的实际与参考加速度差的均方，上限 10000；Stage1 保留双脚/双腕 |
| Action rate | −0.05 | 实际关节目标 `q_target` 的步间差平方和，不是原始 action 差 |
| Joint limit | −15 | 超出 soft joint position limit 的距离和 |
| Undesired contact | −0.1 | 除双脚以外，净接触力范数 >10 N 的刚体数量 |
| Head/torso impact 的当前对应项 | −1e−5 | `torso_link` 净接触力平方和，当前名称为 `auxiliary_anchor_contact_force` |

上表**权重**来自论文；σ、死区、阈值、截断、集合及求和/均值方式主要来自本地 `../PGMT/pgmt/environment.py` / `optimized_v2.yaml` 的工程实现。论文正文不足以证明这些细节与作者训练逐项相同。当前无需为进入 P3 而全面重写既有公式，应保留已训练的基线并单独记录必要偏差。

全局位置修正已完成：[corrected_command()](../pmt_tasks/pgmt_math.py#L81) 在参考 anchor 坐标系计算 XY 位置误差，使用 `gain=0.5`、速度门控 `min(speed/0.5,1)`、逐分量裁剪 ±0.5 m/s；同一目标同时用于 command 和 velocity reward，静止参考不产生位置修正。符合论文公布的修正结构，P3 保留。

## 2. 必须新增的 terrain-contact 组

六项已全部完成计算、缓存、独立 RewardManager 联合验收和正式四头 PPO 接入，共同组成 `terrain` 组。地形容差的 family 开关只约束三项下肢跟踪；flat/rough 上的六项接触奖励仍启用。

| 新奖励 | 论文权重 | 建议计算语义 | 现有可复用资源 |
| --- | ---: | --- | --- |
| Touchdown quality | +10 | 2026-10-18 起双脚实际承重与同帧有效参考标签整体匹配时，每步结算参考支撑脚的平均质量；无支撑或失配为 0 | 实际足位置、地图/扫描有效区域、地形向上力及参考标签；原奖励接口保留 |
| Reference contact match | +1.5 | 左/右脚实际接触与当前帧离线目标匹配；按 `contact_valid` 屏蔽未知标签，全部无效时为 0 | `reference_contacts()`、`robot_foot_contact_forces()` |
| Slip | −1 | 只在实际接触时惩罚脚部切向/平面滑动速度，摆动脚不受此项惩罚 | 足部 link 速度与同序接触力 |
| Stumble | −20 | 惩罚侧向撞击，例如水平力相对竖直力异常偏大 | 接触力；原生 lateral-force 项可参考，但与此判据不等价 |
| Contact switching | −30 | 惩罚短时间内的接触状态反复切换，正常摆动/落脚转换不直接全罚 | 新建 previous contact、状态持续时间及首次测量有效位 |
| Contact force | −1e−6 | 按第 7.6 节 T6 实现，对两脚力范数平方求和，无阈值 | 同序全对象足部接触力 |

论文公开了这些目标与权重，但没有公开完整判定公式及阈值。本地独立实现 [environment.py:315](../../PGMT/pgmt/environment.py#L315) 可作为起点：局部高度方差的 `exp(-var/0.03²)`、接触阈值 10 N、快速切换 0.1 s、绊碰判据 `||Fxy|| > 5|Fz|+10`、脚力超过 350 N 的平方超量。最后一项是本地参考实现的旧候选，本文当前 T6 已采用无阈值平方和。所有未公开细节必须作为**工程设定**集中配置，不能宣称是论文原值，也不应直接整段复制。

接入时需要修正/约束本地独立实现的几个细节：

- **无效标签**：只对有效脚计分，建议 `sum(valid × match) / max(sum(valid),1)`；未知不等于摆动。只对当前参考帧计算，不读取 reset 后的下一段或跨片未来帧。
- **奖励使用干净几何**：不读取加噪、丢点、归一化、裁剪后的策略 `terrain_map`。策略增强变化不应改变真实落脚质量；世界 z 或未处理相对高度均可用于局部方差，但单位必须为 m。
- **采样位置**：使用实际落脚处，不能使用位置选择器预测的四个中心，也不能用整张图方差。无效高度不能先填 −1/0 再算方差；观测范围外或缺少局部有效支持时不发正的落脚质量奖励。缺测不取消接触匹配、slip、stumble 等仍有物理依据的项。
- **事件时序**：接触状态、age、touchdown/switch 每控制步更新一次；21+6 个 RewardManager term 重复访问不能推进状态。reset 初始化接触历史，首次测量不伪造 touchdown 或快速切换；只清理被 reset 的环境。
- **接触与承重分别定义**：2026-10-14 已确定 T1 用脚—地形过滤力的世界 Z 分量 `>10 N` 判断承重，以该支撑状态的上升沿触发；T2 按用户选择保留全对象力范数判据，T3 沿用同一 c，T4 直接比较全对象力的水平/竖直分量，T5 使用 c 的双向快速切换，T6 对全对象脚力直接求平方和。不能把侧撞等接触事件也用承重状态屏蔽。PMT 现有传感器的 10 N 阈值是力范数阈值，`history_length=3、track_air_time` 不会自动变为新的承重判据；[feet_contact_time()](../pmt_tasks/mdp/rewards.py#L256) 只识别短支撑后离地，不能单独替代双向快速切换惩罚。
- **标签几何与仿真几何不同**：离线 TCRS 标签用每脚 4 个球底探针；当前 Isaac USD 每个 ankle-roll body 下有 7 个 capsule collider。两者不能当成同一套碰撞体。接触匹配接入时需统计稳定支撑/正常摆动的标签与实测力差异，再标定阈值；本次不重建标签或改机器人资产。
- **权重与 dt**：沿用当前奖励率约定。新 T1 整体匹配且参考支撑脚平均 quality=1 时，每步贡献为 `10×0.02=0.2`；切换事件一次最多对应每脚 `−30×0.02=−0.6`。权重和 dt 均只乘一次。

对应已有接口：[标签/力/足下地形](../pmt_tasks/pgmt_runtime.py#L106)、[clean scan](../pmt_tasks/pgmt_runtime.py#L148)、[离线标签说明](PGMT_CONTACT_LABELS.zh-CN.md)。接触标签接口已完成不代表接触奖励已经完成。

## 3. 修改三项下肢跟踪：先放宽误差，再进入指数核

2026-10-14 已完成[容差数值核、接口及独立验收](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)：按当前实际 torso 所在 tile 的 family/level，默认采用 L/9、三个既定工程上限及下肢逐元素 α=1；已提供上限、适用地形、等级缩放和 α 配置。误差以 `clamp_min(0)` 保持非负；P2 默认关闭，可显式启用。267 项 CPU/CUDA 联合回归、补充后 30 项容差专项与 Isaac 实际 50 区域检查通过；当时仅验收数值。后续 P3.4 已开启容差并完成四头 PPO 短训与续训。

论文第 5 页：

```text
e_relaxed[h,j] = max(0, e[h,j] − alpha[h,j] × chi(current_family) × tau[h](difficulty))
reward = exp(−mean(e_relaxed²) / sigma[h]²)
```

只作用于 **lower body position / lower body orientation / lower joint position**；默认仅 stairs/slope/boxes 激活，可通过 `terrain_families` 修改地形白名单。上肢、所有速度项继续严格跟踪，默认 flat/rough 严格。不能在平方/均值之后才减容差，也不能通过增大所有 σ 来代替论文的死区结构。

`alpha` 按 link/joint 配置，0 恢复严格跟踪；`tau` 随难度线性增长并饱和。现有计划的上限 `0.20 m / 0.7 rad / 0.5 rad` 可保留为首轮工程初值，并用高度差、跟踪误差分布和任务完成率验证。

**难度来源已在实现中分开。** 论文明确将 family 和 difficulty 定义为机器人当前所在 tile。此前总计划写成了“当前 family + 出生课程等级”，现已修正为奖励取当前位置的 family/level；出生课程等级、升降级和统计仍属于出生分配。跨 tile 后奖励容差跟随实际地形，课程归属不变。当前 manifest 的连续几何 `difficulty` 并不等于 L/9；本轮默认采用 `tile_level/9` 作为首版工程映射，二者不能混用。

对 body-wide 容差，实现使用实际 `torso_link` anchor 的世界 XY，经 `TerrainTileMap.locate()` 查询 tile；双脚跨 tile 的局部落脚质量继续分别查询各脚。torso 与 pelvis/root 不混用，也不使用最近 origin 或一只脚的位置替代该查询点。

## 4. 暂缓的恢复奖励与已有安全项

**Recovery upward velocity（+12.5）按用户要求暂缓。** 当前 `PGMTStage1EnvCfg.pgmt_recovery_enabled=False`，Stage2 继承该值；所用奖励 YAML 的 `recovery_upward_velocity=false`，权重表没有对应项。本轮维持此状态，不单独给普通动作增加向上速度奖励，也不新增 recovery episode、倒地初始化、恢复专属终止宽限或 survival-based 恢复采样。

此前“先平地补恢复再进入地形训练”的前置要求撤销。后续若重新纳入 recovery，再核对本地 `recovery_pool.pt` 的资产、关节顺序、COM/link 速度和适用地形，并独立定义恢复回合的 sampler/课程结算。本轮不要求这些工作完成；普通动作跟踪、接触状态和地形课程继续推进。

**Head/torso impact 的身体归属已经实际碰撞确认。** 实际运行 USD `logs/pgmt_p1_acceptance/native_robot/main.usd` 有 30 个刚体；`/g1/torso_link/head_link` 是非刚体 Xform，torso 下有 4 个 capsule collider（含 torso 局部 z≈0.415 m 的顶部 collider）。头部不是单独的 ContactSensor body。2026-10-14 已通过头/胸各 1/2 m/s 真实碰撞确认两者进入同一 torso 合力项，不额外累加 head；但当前控制步末采样会漏掉短冲击峰值，不能据此声称完整捕获所有冲击强度。见 [原生身体碰撞验收](PGMT_BODY_CONTACT_AUDIT.zh-CN.md)。净力平方与逐碰撞点冲量的定义也不同，论文未给出该细节公式。

`undesired_contacts` 当前惩罚所有非脚接触，包含膝/手。在爬行、起身动作中可能与必要接触冲突；论文仍列此项，但没有公开身体集合或行为掩码。首轮保留原权重和行为，单列动作库中爬行/起身动作的分项统计后再决定是否需要显式行为条件，不能未经记录删项。这里是普通参考动作跟踪，不新增 recovery 回合。pelvis index=0 和末端/impact 集合建议改为名称验证，避免模型更换造成静默错位。

## 5. 四头、终止与迁移必须同步

- [PGMTActorCritic](../motion_tracking_rl/networks/pgmt_actor_critic.py) 已按配置支持三头/四头，新正式任务输出 `[upper, lower, terrain, auxiliary]`，奖励分组同步 `[N,4]`。三项 TA 跟踪仍在 lower；auxiliary 保留 9 项，不加入 recovery。
- 原生 PPO/storage 多头流程已通过四头独立 GAE、分组和、done/timeout、value loss 与梯度验收；各头 value loss 已独立记录。四组优势先求和再归一化，不分别标准化以改变原权重比例。
- [migrate_three_head()](../motion_tracking_rl/utils/pgmt_checkpoint.py) 已实现 Stage1/P2 三头 → 四头：旧 0/1/2 行 → 新 0/1/3 行，terrain 权重/偏置为零，其他旧参数和统计精确复制；新阶段 Adam/iteration/sampler/课程重新初始化。四头 metadata、奖励/容差/感知契约和严格续训已接入；P2 checkpoint 不能被静默当作 P3 同配置续训。
- Stage2 终止已按用户指定修改：root pelvis 的 XY 轨迹误差 >1 m，torso 高度相对于当前位置干净地形＋原动作高度误差 >0.4 m，仅双手腕相对于各自 pelvis 的高度误差 >0.4 m，原投影重力 Z 差 >0.8 保留；超时为 10 s/500 步。前四项连续超限 0.2 s（10 控制步）才终止，恢复正常或 reset 清零；地图越界与动作耗尽继续立即显式截断，详见 [终止说明](PGMT_STAGE2_TERMINATIONS.zh-CN.md)。Stage1 保留原判据。
- timeout bootstrap：[当前 PPO](../motion_tracking_rl/algorithms/ppo.py#L327) 本轮继续沿用原生 `gamma × transition.values`，是采样动作时的旧状态 value，并非截断前最终观测的 value；四头数值测试通过。若后续改为最终观测 bootstrap，必须在 reset 前保存对应观测并同步各头，禁止使用 reset 后新观测。本次短训未自然触发 timeout，相关真实生命周期证据仍参考 P1/P2 专项验收。

## 6. 建议实施顺序与验收

| 步骤 | 工作 | 完成标准 |
| --- | --- | --- |
| P3.1 | **已完成**：奖励配置和数据接口、组顺序、当前几何/出生课程、接触事件/dt、recovery 禁用 | 新正式任务 27 项/四组；Stage1 原 21 项回归通过；Stage2 EE acceleration 按用户要求仅双腕 |
| P3.2 | **奖励容差及用户指定的 Stage2 终止规则已实现** | 三项奖励容差开启；终止使用当前地形支撑高度、相对根部手腕高度和 XY 轨迹，10 s；前四项连续超限 0.2 s 后终止，无额外 reset 宽限，策略性能另验收 |
| P3.3 | **已完成**：六项 terrain-contact 与共享状态、正式四头接入 | 平地/台阶边缘、未知标签、滑动/绊碰/快速切换/脚力、reset 和重复读取通过 |
| P3.4 | **已完成，最新配置补验通过**：四头、显式迁移、配置/metadata/严格续训 | 600 项联合回归，160 环境 3+2 次更新、独立进程二次恢复及 P2 迁移通过；见 P3 总验收 |
| P3.5 | **部分完成**：静态集合与头/胸真实碰撞链路通过；原恢复补课及 +12.5 奖励暂缓 | 5 环境/44 步核对真实 GPU 接触点→sensor→两项惩罚→四组→reset；发现 50 Hz 末子步采样遗漏短冲击峰值，尚未改聚合；爬行膝/手必要接触统计待完成。见 [身体碰撞验收](PGMT_BODY_CONTACT_AUDIT.zh-CN.md) |
| P3.6 | **四头有限短训/保存恢复已通过，完整分级性能验收待完成** | 继续检查各 family/level 完成率、分项回报/事件率及终止适配；不能以本次 5 次更新代替完整评测 |

P3.3 奖励和三项容差已随 P3.4 同时接入正式任务。大规模验收前还需正式接入已定位的 [PhysX 共享地形包围盒修复](PGMT_PHYSX_MEMORY_DIAGNOSIS.zh-CN.md)；上轮 bits=0 仅为诊断对照，不代表训练入口已修复。

原审计阶段验证：Python 权重表与 YAML 的 21 项逐项一致；`tests/test_pgmt_mdp.py` **5 passed**，覆盖当前奖励对本地独立 PGMT 的数值一致性、观测、片段边界、reset 后分组提取和重复读取缓存。该历史结果不替代后续 P3.4 验收，也不是奖励与论文所有未公开细节等价的证明。

## 7. 全部 reward / penalty 的计算公式

公式补充日期：2026-10-13；范围更新：2026-10-14。状态分为：**现有**＝当前 PMT 实际计算；**P3 候选**＝本轮尚未接入程序的具体工程方案；**暂缓**＝保留公式记录但排除本轮实现与验收。论文明确规定奖励名称、权重、地形容差结构及位置修正结构；未公开的核尺度、阈值、局部采样协议和事件初始化方式均为工程设定。记录候选公式不代表已经冻结或启用配置。

### 7.1 符号、坐标系与总奖励

以下省略环境索引，每个公式针对一个环境的控制步 t：

| 符号 | 定义 |
| --- | --- |
| Δt | 控制步长，当前为 0.02 s；不是 0.005 s 的物理子步 |
| 上标 r | 经过本回合统一平移/yaw 变换后的参考；参考关节轨迹保持原值 |
| pᵢ、Qᵢ、vᵢ、ωᵢ | 刚体 i 的世界位置、wxyz 四元数、link 原点线速度、世界角速度；线速度不是 COM 速度 |
| qⱼ、q̇ⱼ | 原始物理单位的关节位置/速度；不使用观测中的 0.05 速度缩放 |
| a、b | anchor `torso_link` 与 base `pelvis`；当前代码 b 使用 body index 0 |
| Rₐ、Yₐ | anchor 的完整旋转矩阵与仅包含 yaw 的旋转矩阵；参考对应 Rₐʳ、Yₐʳ |
| Bᵤ、Bₗ | 上/下肢 link 集合，按 `body_groups()` 中名称筛选 |
| Jᵤ、Jₗ | 上/下肢 joint 集合；Jᵤ 包含 waist，Jₗ 为腿部关节 |
| F、E、D | F＝左右 ankle-roll；E＝双脚加双 wrist-yaw；D＝所有刚体减去 F |
| Fᵢ | 当前控制步读取的刚体世界法向接触力合力 `net_forces_w`，单位 N；当前版本不含摩擦力，现有全身传感器也未按地形过滤；并非把所有接触点的力范数逐个相加 |
| qᵈᵉˢ | articulation 实际关节位置目标；对应 `robot.data.joint_pos_target` |
| [x]₊、1[条件] | `max(x,0)` 与条件指示量；`clip(x,l,u)` 对每个分量裁剪 |

定义未加权项为 fᵢ，加权奖励率为 ρᵢ。下文每个 ρ 公式**已经包含正/负权重**，不能再乘一次表中权重。

$$
\rho_{t,i}=w_i f_{t,i},\qquad
r_t^{g}=\Delta t\sum_{i\in g}\rho_{t,i},\qquad
r_t=\sum_g r_t^g.
$$

当前组为 `[upper, lower, auxiliary]`，且没有 recovery 项；本轮 P3 目标为 `[upper, lower, terrain, auxiliary]`、共 27 项，auxiliary 求和排除 A4。RewardManager 接口仍返回 fᵢ，由 manager 乘 wᵢ、Δt；这里的 ρ 仅用于公式表达。负权重的 penalty 对应非负 fᵢ，故其 ρ≤0。

以下是环境奖励；PPO 的 entropy bonus、value loss、KL 调节和终止判断不计入这 28 项，也不凭空增加 survival reward 或 termination penalty。

### 7.2 十二项身体/关节跟踪：误差定义

位置和姿态先作 anchor/yaw 对齐：

$$
\bar p_i=Y_a^\top(p_i-p_a),\qquad
\bar p_i^r=(Y_a^r)^\top(p_i^r-p_a^r),
$$

$$
\bar Q_i=Q_{\mathrm{yaw},a}^{*}\otimes Q_i,\qquad
\bar Q_i^r=(Q_{\mathrm{yaw},a}^{r})^{*}\otimes Q_i^r.
$$

`*` 表示单位四元数共轭。当前 `angle()` 的数值定义为：令 `δQ=Q₁⊗Q₂*=(δw,δv)`，则

$$
\operatorname{ang}(Q_1,Q_2)
=2\operatorname{atan2}\!\left(\|\delta v\|_2,
\max(|\delta w|,10^{-9})\right).
$$

六种逐元素误差为：

$$
\begin{aligned}
e_{p,i}&=\|\bar p_i-\bar p_i^r\|_2,\\
e_{Q,i}&=\operatorname{ang}(\bar Q_i,\bar Q_i^r),\\
e_{v,i}&=\|Y_a^\top v_i-(Y_a^r)^\top v_i^r\|_2,\\
e_{\omega,i}&=\|Y_a^\top\omega_i-(Y_a^r)^\top\omega_i^r\|_2,\\
e_{q,j}&=|q_j-q_j^r|,\\
e_{\dot q,j}&=|\dot q_j-\dot q_j^r|.
\end{aligned}
$$

注意：body velocity 项只旋转世界速度，**没有减去 anchor 速度**。位置/线速度/角速度先对 XYZ 求范数，再在 body 集合内对平方误差求均值；关节位置不做额外周期角 wrap。

对元素集合 S，定义：

$$
\mathcal E_S(e,\sigma)
=\exp\!\left(-\frac{1}{|S|\sigma^2}\sum_{j\in S}e_j^2\right).
$$

### 7.3 十二项跟踪奖励的逐项公式

当前所有下肢误差严格计算，即下表的 `ẽ=e`。P3 只替换 L1/L2/L5 的误差，权重及 σ 不变。

| ID | 配置项 | 加权奖励率 ρ（现有权重） |
| --- | --- | --- |
| U1 | `upper_body_position` | $\mathcal E_{B_u}(e_p,0.15)$ |
| U2 | `upper_body_orientation` | $\mathcal E_{B_u}(e_Q,0.4)$ |
| U3 | `upper_body_linear_velocity` | $0.5\,\mathcal E_{B_u}(e_v,1.0)$ |
| U4 | `upper_body_angular_velocity` | $0.5\,\mathcal E_{B_u}(e_\omega,3.14)$ |
| U5 | `upper_joint_position` | $\mathcal E_{J_u}(e_q,0.4)$ |
| U6 | `upper_joint_velocity` | $0.5\,\mathcal E_{J_u}(e_{\dot q},5.0)$ |
| L1 | `lower_body_position` | $0.5\,\mathcal E_{B_l}(\tilde e_p,0.15)$ |
| L2 | `lower_body_orientation` | $2.0\,\mathcal E_{B_l}(\tilde e_Q,0.4)$ |
| L3 | `lower_body_linear_velocity` | $0.5\,\mathcal E_{B_l}(e_v,1.0)$ |
| L4 | `lower_body_angular_velocity` | $0.5\,\mathcal E_{B_l}(e_\omega,3.14)$ |
| L5 | `lower_joint_position` | $0.5\,\mathcal E_{J_l}(\tilde e_q,0.4)$ |
| L6 | `lower_joint_velocity` | $0.5\,\mathcal E_{J_l}(e_{\dot q},5.0)$ |

已实现的 P3 容差（下式为默认地形/等级配置，原 P2 关闭，新四头任务开启）：

$$
\tilde e_{h,j}
=\left[e_{h,j}-\alpha_{h,j}\,\chi(\kappa_t)\,\tau_h(L_t)\right]_+,
\quad h\in\{p,Q,q\},
$$

$$
\chi(\kappa)=\mathbf1[\kappa\in\{\mathrm{stairs,slope,boxes}\}],\qquad
\tau_h(L)=m_h\,\operatorname{clip}(L/9,0,1).
$$

首版默认初值 `mₚ=0.20 m、mQ=0.7 rad、mq=0.5 rad`，下肢 12 个 link/12 个 joint 的 `α=1`，其余目标不放宽；允许按名称逐元素覆写。这些基础上限及 `L/9` 是工程设定，实际元素上限为 αm。κₜ、Lₜ 取实际 torso anchor 所在 tile，出生等级另用于课程。配置入口、代码和结果见[容差实施与验收](PGMT_TRACKING_RELAXATION_PLAN.zh-CN.md)。没有有效 tile 时，已实现零容差结算并明确记录无效，不把越界伪装成有效地形；地图越界仍按既有规则截断。

容差先作用于每个原始误差，再平方、求均值；不改变上肢、L3/L4/L6 或 auxiliary 项。

### 7.4 Auxiliary 奖励与惩罚（9 项本轮范围 + 1 项暂缓）

**A1 — Root orientation（现有，+0.5）**

$$
\rho_{\mathrm{root\_ori}}
=0.5\exp\!\left(-\operatorname{ang}(Q_a,Q_a^r)^2/0.4^2\right).
$$

对应 `auxiliary_root_orientation`。这里比较 torso 完整世界姿态，包含 yaw；不能替换成上面的 yaw 消除后的 link 姿态误差。

**A2 — Corrected root velocity（现有，+2.0）**

先在参考 anchor **完整旋转**坐标系计算名义速度、实际速度和位置误差：

$$
u^r=[(R_a^r)^\top v_a^r]_{xy},\quad
u=[(R_a^r)^\top v_a]_{xy},\quad
e^{\mathrm{anchor}}=[(R_a^r)^\top(p_a^r-p_a)]_{xy}.
$$

$$
g=\operatorname{clip}(\|u^r\|_2/0.5,0,1),\qquad
\tilde u^r=u^r+\operatorname{clip}(0.5\,g\,e^{\mathrm{anchor}},-0.5,0.5),
$$

$$
\rho_{\mathrm{root\_vel}}
=2.0\exp\!\left(-\frac{\|u-\tilde u^r\|_2^2}{2\times1.0^2}\right).
$$

对应 `auxiliary_root_velocity`。分母中的 **2** 来自 XY 两个分量的 `mean`，不是范数平方直接除以 σ²。修正逐分量裁剪，单位 m/s；同一修正用于 command 和 reward；名义速度为零时 g=0。此处不奖励竖直速度跟踪。

**A3 — Floating-anchor position（现有，+1.0）**

$$
d_{xy}=\|[p_a-p_a^r]_{xy}\|_2,\qquad
\rho_{\mathrm{anchor\_pos}}
=\exp\!\left(-[d_{xy}-0.3]_+^2/0.5^2\right).
$$

对应 `auxiliary_anchor_position`；使用世界 XY 距离，不包含 Z。0.3 m 内为满分，超出后衰减。

**A4 — Recovery upward velocity（暂缓，+12.5 仅保留对照）**

以下为后续参考公式，本轮不实现、不启用、不纳入 auxiliary 求和。

$$
\rho_{\mathrm{recovery}}
=12.5\,\mathbf1[\mathrm{recovery\ episode}]
\,\mathbf1[p_{b,z}-z_{\mathrm{support}}<0.5]
\,\operatorname{clip}(v_{b,z},0,1.0).
$$

保留候选项名 `auxiliary_recovery_upward_velocity`。低高度阈值 0.5 m、速度上限 1 m/s 来自本地独立实现，非论文公开常数。若后续恢复此任务范围，候选方案仅在兼容平地恢复回合启用，`z_support` 为该平地真实世界高度（不是无条件 0）；在此条件下与本地实现的 `env_origins[:,2]` 口径对应。普通回合、已超过高度门槛、向下运动均不给该奖励。非平地恢复的支撑高度定义尚未冻结，不按此式直接推广；当前程序没有该奖励或 recovery 回合，本轮也不新增。

后续两项使用 link 线速度差分：

$$
a_{t,i}=\frac{v_{t,i}-v_{t-1,i}}{\Delta t},\qquad
a^r_{t,i}=\frac{v^r_{t,i}-v^r_{t-1,i}}{\Delta t}.
$$

**A5 — Pelvis vertical acceleration（现有，−0.001）**

$$
\rho_{\mathrm{base\_acc}}
=-10^{-3}\min(a_{t,b,z}^2,10000).
$$

对应 `auxiliary_base_acceleration`。只惩罚 pelvis 世界竖直加速度；没有减去参考 pelvis 加速度，没有使用躯干角加速度。

**A6 — End-effector acceleration mismatch（现有，−0.001）**

$$
\rho_{\mathrm{ee\_acc}}
=-10^{-3}\min\!\left(
\frac{1}{3|E|}\sum_{i\in E}\|a_{t,i}-a^r_{t,i}\|_2^2,10000\right).
$$

对应 `auxiliary_end_effector_acceleration`。**Stage2 的 E 仅为双腕**：`left_wrist_yaw_link`、`right_wrist_yaw_link`，共 2 个 link；同时对双腕与 XYZ 分量取均值，因此分母为 **6**。脚踝的实际或参考加速度不参与该项。Stage1 保留双脚/双腕，共 4 个 link、分母为 12。先均值后截断，不是逐末端截断后求和；权重仍为 −0.001，上限仍为 10000，由 RewardManager 乘一次 dt。reset 后用新回合状态初始化差分历史，避免把瞬移算成加速度。

选择接口为 `PGMTPerceptionEnvCfg.pgmt_ee_acceleration_body_names`，`pgmt_stage2` 和四头 `pgmt_stage2_terrain` 均生效；`PGMTState` 按完整机器人 body 名称解析索引，其他 body 分组和奖励不受影响。该字段写入环境配置和 checkpoint 的训练契约，因此修改前的 Stage2 checkpoint 不可直接作为同配置严格续训。

验证：45 项 CPU/CUDA 回归通过，覆盖脚部加速度排除、双腕六分量均值、参考加速度抵消、上限、body 顺序、Stage1 数值一致性及 reset 缓存，见 [测试输出](../logs/pgmt_wrist_acceleration/cpu_cuda_tests.txt)。headless Isaac 实际构建三个任务配置，确认 Stage1 四末端、两个 Stage2 双腕及配置序列化，见 [配置核验](../logs/pgmt_wrist_acceleration/native_config_check.json)。本次未运行新的训练；配置核验结束后 Isaac 已退出。核验脚本初次遇到 Isaac 模块循环导入，沿用正式入口先导入 `isaaclab.envs` 后通过，无需更改生产训练入口。

**A7 — Action rate（现有，−0.05）**

$$
\rho_{\mathrm{action\_rate}}
=-0.05\sum_{j=1}^{29}(q^{\mathrm{des}}_{t,j}-q^{\mathrm{des}}_{t-1,j})^2.
$$

对应 `auxiliary_action_rate`。读取实际关节目标；不使用原始 policy action，**不除以 Δt 或 Δt²**，也不对 29 个关节取均值。

**A8 — Joint limit（现有，−15）**

$$
\rho_{\mathrm{joint\_limit}}
=-15\sum_{j=1}^{29}\left([q_j^{\min}-q_j]_+ + [q_j-q_j^{\max}]_+\right).
$$

对应 `auxiliary_joint_limit`，上下限来自运行时 `soft_joint_pos_limits`。这是超限距离的一次和，不平方；正常范围内为零。

**A9 — Undesired contact（现有，−0.1）**

$$
\rho_{\mathrm{undesired}}
=-0.1\sum_{i\in D}\mathbf1[\|F_i\|_2>10\ \mathrm N].
$$

对应 `auxiliary_undesired_contacts`。D 排除双脚，其余刚体均计数；对每个刚体计一次，不按碰撞点数量计数。目前不因爬行/恢复动作而屏蔽膝/手。

**A10 — Head/torso impact 对应项（现有，−1e−5）**

$$
\rho_{\mathrm{impact}}=-10^{-5}\|F_a\|_2^2.
$$

对应 `auxiliary_anchor_contact_force`，a 为 torso；没有阈值或额外裁剪，不乘碰撞时间形成 impulse。头部按第 4 节的实际刚体归属审计；当前公式只有一个 torso 净力，不额外累加名为 head 的节点，也不能写成尚未实现的多身体力求和。

### 7.5 Terrain-contact 的接触状态与事件定义（已实现基础状态）

本节基础状态和事件已于 2026-10-14 接入 Stage2；下节 T1–T6 已完成数值核、独立联验和正式四头 PPO 接入。实现以本地独立方案为起点，补充无效掩码和 reset 语义，将 T1 的承重状态 s 与接触状态 c 分开。用户已明确要求 T2 保留全对象法向力合力范数定义，T3 沿用该 c，T4 直接使用全对象力分量，T5 使用 c 的双向快速切换，T6 无阈值地累计全对象脚力平方；基础状态和 T5 阈值集中于 `CONTACT_STATE_CONFIG`，T4 参数集中于 `STUMBLE_CONFIG`：

$$
c_{t,f}=\mathbf1[\|F_{t,f}\|_2>10\ \mathrm N],\qquad f\in F.
$$

**T1 已确定的承重判据：只看脚与地形之间作用力的世界 Z 正分量是否超过 10 N。** 令 G 为训练地形的实际碰撞体集合，Fᵀ 为其对该脚的法向接触力合力：

$$
F^T_{t,f}=\sum_{g\in G}F^{\mathrm{normal},w}_{t,f\leftarrow g},\qquad
s_{t,f}=\mathbf1[F^T_{t,f,z}>10\ \mathrm N].
$$

10 N 沿用当前讨论阈值；比较的是世界竖直分量，既不取三维范数，也不取 `abs(F_z)`，更不使用随脚旋转的局部 Z。纯水平接触力和向下作用力不产生承重状态。Fᵀ 使用当前 Isaac 接触传感器提供的法向力口径，不隐式加入尚未启用的摩擦力通道，因此是本项目的承重判据，并非完整地面反力或静态平衡证明。

**已实现的过滤与接口：** 左右脚分别建立独立的原生接触传感器，显式列出所有有效训练 tile/patch 的实际碰撞 prim，再对 `force_matrix_w` 的地形对象维求和。v6 共 800 个碰撞体，运行时与导入 USD 逐项核对；不能用全局通配表达式替代显式列表。按实际接触对象统计，不只过滤出生 tile；排除机器人、其他对象和外边界地形。[PGMTState](../pmt_tasks/pgmt_runtime.py) 的 `robot_foot_contact_forces()` 返回全对象法向力合力，`robot_foot_terrain_forces()` 返回地形过滤力，`contact_state(for_reward=True)` 返回本步冻结数据。缺少过滤传感器或结构错误直接报错，不回退到未过滤力；单脚力出现 NaN/Inf 时仅中断对应通道历史，再次有效时重新建基线，不把缺测当成确认未承重。T1 数值核已接入共享事件，并在新正式四头任务启用。

本次确定的是向上力承重判据；此前讨论的 10/5 N 双阈值滞回、足底近地约束仍未加入。参考标签记为 cʳₜ,𝒻；其有效性记为 mʳₜ,𝒻。参考有效性只控制 T2，不控制 T1 的实际承重事件或其他物理接触惩罚。

令 bᶜₜ,𝒻、bˢₜ,𝒻 分别表示接触/承重通道的前后两份测量有效、来自同一回合且已有历史；两个通道独立处理缺测。Aₜ₋₁,𝒻 是上一控制步保存的接触状态年龄，nₜ 是本回合自 1 起计的控制步数。以下事件仅在对应有效测量上定义，无效时事件为 0：

$$
D_{t,f}=b^s_{t,f}\,s_{t,f}(1-s_{t-1,f}),\qquad
S_{t,f}=b^c_{t,f}\,\mathbf1[c_{t,f}\ne c_{t-1,f}],
$$

$$
K_{t,f}=S_{t,f}\,\mathbf1[A_{t-1,f}<0.1\ \mathrm s]\,\mathbf1[n_t>5].
$$

D 为新的承重上升沿，S 为 c 的任一方向接触切换，K 为 c 的快速切换。T1 另用下节的连续未承重门槛得到有效落脚事件 Dᑫ；该门槛不改变其他项的 c/S/K。`n>5` 沿用本地实现的初始切换保护，不代表所有奖励停用；也不替代恢复终止宽限。

每步先用旧年龄生成事件，再更新一次年龄并冻结供奖励读取：

$$
A_{t,f}=\begin{cases}
0,& b^c_{t,f}=0\ \text{或}\ S_{t,f}=1,\\
A_{t-1,f}+\Delta t,& \text{其他情况}.
\end{cases}
$$

随后分别保存 cₜ,𝒻 与 sₜ,𝒻 及各自有效性。这是“自首次观测/最近接触切换后累计”的离散年龄口径；切换当步记 0，K 读取更新前的年龄。每次 reset 对该环境清除两种历史和年龄；首次测量只建立基线，不生成 D/S/K。持续承重不重复生成 touchdown 事件，RewardManager 多次读取同一步也不重复推进 age。原始状态、事件、计时、脚位姿/速度和参考 motion/frame/标签一起冻结；终止 reset 不覆盖旧回合奖励快照。字段与连续时长口径见 [接口说明](PGMT_CONTACT_STATE.zh-CN.md)。

### 7.6 六项 terrain-contact 的逐项公式（T1–T6 均已实现并独立联验）

**T1 — Touchdown quality（+10，计算、独立验收及正式四头 PPO 接入已完成）**

**当前规则（2026-10-18，`support_pattern_mean_v2`）：双脚整体支撑匹配门控，持续结算。**

令 `v_ref` 还包含标签为有限二值的检查，`s_f=1[F_terrain,f,z>10 N]`，`Q_f` 为下文定义的原始九点质量：

$$
M_t=\prod_{f\in\{L,R\}}v^s_{t,f}v^{ref}_{t,f}\mathbf1[s_{t,f}=c^{ref}_{t,f}],\qquad
N^{ref}_t=\sum_f v^{ref}_{t,f}\mathbf1[c^{ref}_{t,f}=1],
$$

$$
\rho_{T1}=10M_t\frac{\sum_f\mathbf1[c^{ref}_{t,f}=1]Q_{t,f}}{\max(1,N^{ref}_t)},\qquad
r_{T1}=\Delta t\,\rho_{T1}.
$$

两脚标签和力测量都有效且状态完全一致才允许结算；任一脚多支撑、少支撑或未知，整体为 0。双脚摆动为 0。只对整体匹配环境中的支撑脚查询当前九点；某脚几何无效时该脚质量为 0，分母仍按参考支撑脚数计。单支撑、双支撑总上限均为 **0.2/步**（dt=0.02）。首次真实测量、稳定支撑和重新接触使用同一个持续公式，落脚不再额外加分。原 `terrain_touchdown_quality`、`pgmt_reward()` 和 `touchdown_state()` 保持，新增门控诊断字段。完整接口、契约版本和验收见 [T1 文档](PGMT_TOUCHDOWN_QUALITY.zh-CN.md)。

**以下为历史事件版推导（2026-10-14），仅对应旧训练和旧验收，不再是当前 T1 结算规则。** 九点几何和方差公式仍被复用，事件门控、两脚求和及 0.4 上限已由上述公式替代。

**激活方式（2026-10-14 更新）：每只脚仅在一次有效落脚的控制步发放一次。** 使用第 7.5 节的承重状态 `s=1[F^T_z>10 N]`，Fᵀ 只统计地形对该脚的作用力；不以参考动作的支撑标签触发。原始事件 D 要求同回合已初始化、上一控制步未承重、本控制步承重。

为避免单步受力丢失/阈值抖动反复领取正奖励，已实现的事件门槛保留此前的 **2 个控制步**，要求连续未承重 `s=0`；50 Hz 下按离散步数累计为 0.04 s。未承重不等于几何上已经离地，例如脚轻触地面或仅侧向碰撞也可能为 `s=0`。这是工程门槛，不是论文参数，不以未承重时间长短额外放大奖励。令 ℓₜ₋₁,𝒻 为上一控制步结算后保存的连续未承重步数：

$$
D^Q_{t,f}=D_{t,f}\,\mathbf1[\ell_{t-1,f}\ge2].
$$

用旧计数计算本步事件后统一更新，令 vˢ 表示当前承重力测量有效：

$$
\ell_{t,f}=\begin{cases}
0,&v^s_{t,f}=0\ \text{或}\ s_{t,f}=1,\\
1,&v^s_{t,f}=1\ \text{且}\ b^s_{t,f}=0\ \text{且}\ s_{t,f}=0,\\
\ell_{t-1,f}+1,&b^s_{t,f}=1\ \text{且}\ s_{t,f}=0.
\end{cases}
$$

每个环境/每只脚独立保存 `[num_envs,2]` 支撑历史和未承重步数。reset 时清理该环境的 `bˢ、ℓ`；新回合首次支撑读数仅建立基线。若首次读数为未承重，从该读数计第一步，以后满足未承重门槛后的落脚可正常触发。这与 K 使用的 c 状态年龄 A 是两个不同计数口径。Dᑫ 已通过 `touchdown` 字段提供，下表行为已在 T1 数值核及独立 RewardManager 配置中实现并验收。

| 情形 | T1 行为 |
| --- | --- |
| 连续未承重达到门槛，随后首次承重 | 用该控制步的实际脚位置/yaw 查询 9 点，只结算一次 |
| 持续站立、持续支撑、保持承重的滑动 | 为 0，不重复发放 |
| 仅一个控制步判为未承重后重新承重 | 不发 T1；其他接触/切换项仍按各自定义计算 |
| 纯水平侧撞，或作用力来自机器人而非地形 | 不产生 T1 承重上升沿 |
| reset 后首次检测到承重 | 为 0，避免把初始化支撑算成落脚 |
| 落脚当步 9 点未全部有效 | 为 0，该落脚事件仍被消耗；后续数据有效不补发 |
| 左右脚同时满足有效落脚 | 分别计算质量并求和 |

已在 [PGMTState.rewards()](../pmt_tasks/pgmt_runtime.py) 的一次性结算中生成事件并冻结旧/新状态，仅对 Dᑫ 为真的脚批量查询地形，缓存 `terrain_touchdown_quality` 未加权率。无事件时 T1 直接为 0；重复 RewardManager term 读取同一步不再射线查询或更新计数。旧回合奖励在 done/reset 前结算，reset 只初始化下一回合历史，不回写本步已结算奖励。触发时刻采用 50 Hz 控制步的接触读数，不宣称捕获控制步内全部 200 Hz 短暂碰撞；尚未新增 substep 事件锁存。

建议项名 `terrain_touchdown_quality`。2026-10-13 讨论已确定采样窗口：以实际脚踝 link 为基准，中心沿脚尖方向前移 **0.039 m**，纵向保持 **0.20 m**，横向改为 **0.10 m**，仍取 **3×3 点**。纵向点距为 0.10 m，横向点距为 0.05 m。

尺寸依据：当前训练 USD 每脚 7 个胶囊的局部几何外包范围约为 `X=[−0.064,0.142] m、Y=[−0.036,0.036] m`，长 0.206 m、最大宽 0.072 m，XY 外包中心为 `(0.039,0) m`。左右脚相同；这里是碰撞几何外包范围，不是同时承重的接触面积。来源：[训练 URDF](../pmt_tasks/assets/unitree_description/urdf/g1/main.urdf#L191)、[胶囊转换配置](../pmt_tasks/robots/g1.py#L75)，并已读取 `logs/pgmt_p1_acceptance/native_robot/main.usd` 核对。

实现使用**每只实际脚自身的 yaw**，记对应旋转为 Y𝒻。以下定义的是水平采样模板，暂不按脚的 roll/pitch 压缩投影；这是已记录的工程选择：

$$
x_{t,f,k}=[p_{t,f}]_{xy}+[Y_f(d+\delta_k)]_{xy},\qquad d=(0.039,0,0),
$$

$$
\delta_k\in\{-0.10,0,0.10\}\times\{-0.05,0,0.05\}\times\{0\}.
$$

因此脚局部的 9 个 XY 坐标是 `X∈{−0.061,0.039,0.139} m` 与 `Y∈{−0.05,0,0.05} m` 的笛卡尔积；先加局部偏移再随脚旋转，不能固定沿世界 +X 平移。

**已实现的高度获取：直接复用真实地形网格射线查询。** 仅收集本步落脚的脚，将每脚 9 点展平后调用 [PGMTTerrainImporter.query()](../pmt_tasks/pgmt_terrain_importer.py#L112)，再将 `height` 与 `valid` 恢复为 `[num_envs,2,9]`。底层 [height()](../pmt_tasks/pgmt_terrain_importer.py#L87) 从高于整张地形最高点 5 m 的 `ray_top` 沿世界 −Z 发射射线，取地形三角网格第一个交点的世界 Z；查询网格与现有扫描共享，射线不包含机器人。查询按每个点的实际位置进行，跨 tile 不沿用出生 tile 的高度。

zₜ,𝒻,ₖ 为这些位置的干净世界地面高度（m），保留负高度。当前 21×21 策略扫描的网格间距为 0.10 m，不能直接给出横向间隔 0.05 m 的真实高度；插值也可能把台阶边缘变成连续斜面，因此实现不从策略 `terrain_map` 插值，不使用策略选择器中心、加噪、dropout、归一化或填充值。

`query()` 已合并地图范围与射线命中有效性；无效点返回 `valid=False、height=NaN`。沿用本节原有的扫描覆盖限制时，再按当前扫描器的位置/yaw 将查询点变换到扫描坐标，检查 XY 是否在 `[-1,1]×[-1,1] m` 内。这是几何覆盖掩码，与策略随机 dropout 无关。只保留在有效地图内、扫描覆盖内且高度有限的样本集合 Vₜ,𝒻。若 N=|V|≥1，计算总体方差：

$$
\mu_{t,f}=\frac1N\sum_{k\in V_{t,f}}z_{t,f,k},\qquad
s^2_{t,f}=\frac1N\sum_{k\in V_{t,f}}(z_{t,f,k}-\mu_{t,f})^2.
$$

已实现的首版要求 9 点全部有效才发该项正奖励：

$$
Q_{t,f}=\begin{cases}
\exp(-s_{t,f}^2/0.03^2),&N=9,\\
0,&N<9,
\end{cases}
\qquad
\rho_{\mathrm{touchdown}}=10\sum_{f\in F}D^Q_{t,f}Q_{t,f}.
$$

每步实际贡献为 `r_touchdown=Δt×ρ_touchdown`，权重和 dt 都只乘一次。单脚在 9 点等高且有效的落脚当步最多得 `10×0.02=0.2`，双脚同时落脚最多 0.4；下一步持续支撑为 0。窗口尺寸、中心偏移、向上力承重判据与落脚时单次激活已按上述讨论更新；连续未承重门槛、高度查询方案、脚 yaw 模板、全有效门槛和 0.03 m 尺度已作为首版工程配置实现，论文仅要求用局部高度变化评价落脚。方差分母是 N，等价 `unbiased=False`，不是样本方差的 N−1；不要再次平方 s²。无有效样本时直接返回 Q=0，不计算空集方差，也不通过 `NaN×0` 屏蔽。常数高度偏移不改变质量；不能用增强噪声、padding 或归一化数值代入。

**T2 — Reference contact match（+1.5，计算、独立验收及正式四头 PPO 接入已完成）**

2026-10-14 用户确认：实际状态使用全对象接触力范数 `c=1[||F_net_normal_world||₂>10 N]`，不替换为 T1 的地形向上力承重。令 `vᶜ` 为实际接触力测量有效性：

$$
\rho_{\mathrm{contact\_match}}
=1.5\frac{\sum_{f\in F}m^r_{t,f}v^c_{t,f}\,\mathbf1[c_{t,f}=c^r_{t,f}]}
{\max(\sum_{f\in F}m^r_{t,f},1)}.
$$

实现项名 `terrain_reference_contact_match`。有效参考支撑、摆动均参与二值匹配；两个参考标签均无效时为 0，仅一脚参考有效时只评价该脚。实际测量无效的脚不获得匹配分，分母仍为有效参考数量，避免缺测提高得分。此式是对本地独立实现“两脚直接 mean”的掩码补充，不将 unknown 视为 swing，也不自动视为匹配成功。

T2 每控制步持续结算，首次有效测量即可评价，无 touchdown/历史门槛。`PGMTState.reference_contact_match_state(for_reward=True)` 提供同一旧回合参考帧上的匹配结果和掩码；reset 只清理当前副本，终止快照保留。权重与 dt 仅由 RewardManager 乘一次；两脚有效且全匹配时 `1.5×0.02=0.03`，仅一脚匹配时为 0.015。107 项 CPU/CUDA 回归和 5 环境/72 步原生 RewardManager 验收通过，详见 [T2 记录](PGMT_REFERENCE_CONTACT_MATCH.zh-CN.md)。

**T3 — Slip（−1.0，计算、独立验收及正式四头 PPO 接入已完成）**

$$
\rho_{\mathrm{slip}}
=-\sum_{f\in F}c_{t,f}\,\|[v_{t,f}]_{xy}\|_2^2.
$$

实现项名 `terrain_slip`。沿用本地实现的**世界 XY link 原点速度**，两脚求和；不是相对参考脚速度，也不是脚底接触点速度。斜坡表面切向投影是另一种可选定义，当前没有默认为该形式；如采用需额外查询法向并重新记录公式。没有实际接触的摆动脚不受此项惩罚。

实际门控使用与 T2 相同的全对象接触状态 c，参考标签和 touchdown 不参与门控；首次有效测量和持续接触均计算。数值核返回非负代价，RewardManager 乘 −1.0 与 dt 各一次，没有速度死区、额外裁剪或两脚平均。一只接触脚水平速度 0.5 m/s 时，当前单步贡献为 −0.005。异常测量通过有效性标记隔离，不能把未知数据当作有效无滑移。

`PGMTState.slip_state(for_reward=True)` 冻结同一旧回合的速度、接触、逐脚代价和时间信息；局部 reset 保留终止快照。125 项 CPU/CUDA 回归、5 环境/72 步原生 RewardManager 验收通过；另由质心速度及角速度独立重建 link 速度，确认输入定义。详见 [T3 实现与验收](PGMT_SLIP.zh-CN.md)。

**T4 — Stumble（−20，计算、独立验收及正式四头 PPO 接入已完成）**

$$
\rho_{\mathrm{stumble}}
=-20\sum_{f\in F}\mathbf1[\|[F_{t,f}]_{xy}\|_2>5|F_{t,f,z}|+10\ \mathrm N].
$$

项名 `terrain_stumble`。使用冻结的全对象世界法向接触力合力，严格 `>`，正负 Fz 均取绝对值；逐脚二值判断后求和。每控制步持续计罚，首次有效测量即可计算，不乘参考支撑标签、地形承重、touchdown 或历史门槛。比例 5 与偏置 10 N 集中于 `STUMBLE_CONFIG`，属于工程初值。

数值核返回 0/1/2 非负代价率，RewardManager 乘 −20 和 dt 各一次；当前单脚触发每步 −0.4，双脚 −0.8。异常力与运算溢出明确标记无效并隔离，不当作有效无绊碰。`PGMTState.stumble_state(for_reward=True)` 提供力、阈值、触发标记和时间信息；终止快照保留，局部 reset 清空当前副本。164 项 CPU/CUDA 回归和 5 环境/102 步 Isaac 通过；真实侧撞 2 次触发，NumPy 代价率误差为 0。详见 [T4 实现与验收](PGMT_STUMBLE.zh-CN.md)。

**T5 — Contact switching（−30，计算、独立验收及正式四头 PPO 接入已完成）**

$$
\rho_{\mathrm{switching}}=-30\sum_{f\in F}K_{t,f}.
$$

项名 `terrain_contact_switching`。复用第 7.5 节的 `fast_contact_switch/contact_history_valid`，全对象接触 c 的两个切换方向均检查；旧年龄严格 `<0.1 s` 且本回合步数 `>5` 时触发。数值核不重复维护计时器，也不使用归零后的新年龄或 `contact_time/air_time`；恰好 0.1 s 不计罚。只在切换当步计罚，保持状态不重复发放，参考标签与地形承重不参与门控。

数值核返回非负率 0/1/2，RewardManager 乘 −30 和 dt 各一次，当前单脚触发 −0.6、双脚 −1.2。`PGMTState.contact_switching_state(for_reward=True)` 保留本步旧回合事件、前后状态/年龄和时间信息；reset 清理当前副本，首次有效测量和缺测恢复不产生伪切换。195 项 CPU/CUDA 回归及 5 环境/102 步 Isaac 验收通过；由原始力独立重放历史，4 次快速切换（进入/退出各 2 次）累计约 −2.4，代价率误差为 0。详见 [T5 实现与验收](PGMT_CONTACT_SWITCHING.zh-CN.md)。

**T6 — Contact force（−1e−6，计算、独立验收及正式四头 PPO 接入已完成）**

$$
\rho_{\mathrm{foot\_force}}
=-10^{-6}\sum_{f\in F}[\|F_{t,f}\|_2]^2.
$$

项名 `terrain_contact_force`。对左右脚的全对象世界法向接触力 XYZ 三分量平方逐脚求和，不减去 350 N，不设上限，也不乘 10 N 接触状态 c。首次有效测量和每个后续控制步均计算，参考标签、地形承重及事件不参与门控；两脚合力方向相反不会抵消本项代价。与 A10 的 torso 无阈值净力平方采用不同身体集合和权重，A10 保持原状。

数值核返回非负代价率，RewardManager 乘 −1e−6 和 dt 各一次；当前单脚 450 N 贡献 −0.00405，单脚 5 N 仍贡献 −5×10⁻⁷。缺测、非有限力或平方/总和溢出明确标为无效并隔离。`PGMTState.contact_force_state(for_reward=True)` 保留旧回合力、平方代价和时间信息，局部 reset 清空当前副本。223 项 CPU/CUDA 回归和 5 环境/102 步 Isaac 通过；真实力平方的 NumPy 最大相对误差约 1.15×10⁻⁷，六项以 27 项/四组独立配置联验通过。详见 [T6 实现与验收](PGMT_CONTACT_FORCE.zh-CN.md)。

### 7.7 组装、边界行为与核对例子

$$
\begin{aligned}
r_t^{\mathrm{upper}}&=\Delta t\sum_{i=1}^{6}\rho_{U_i},\\
r_t^{\mathrm{lower}}&=\Delta t\sum_{i=1}^{6}\rho_{L_i},\\
r_t^{\mathrm{terrain}}&=\Delta t\sum_{i=1}^{6}\rho_{T_i},\\
r_t^{\mathrm{auxiliary}}&=\Delta t\sum_{i\in\{1,2,3,5,6,7,8,9,10\}}\rho_{A_i}.
\end{aligned}
$$

以上组装式已在新正式任务 `pgmt_stage2_terrain` 生效，共 27 项、排除暂缓的 A4；T1–T6 已进入 terrain 组，L1/L2/L5 默认使用地形容差。原 P2 三头任务仍采用 21 项和严格误差。所有奖励使用同一结算时刻的旧回合状态、参考帧和接触数据。实际物理状态/接触力出现非有限值应由错误保护处理，不能把其简单置零后继续宣称奖励正常。

以下是公式量纲和归约核对例子，不是本次新增的运行测试：

- 仅原 21 项完美静止跟踪、无惩罚：upper 率=4.5、lower 率=4.5、auxiliary 率=3.5，三组总率=12.5，单步贡献=0.25；新任务另叠加 terrain 组。
- root XY 速度误差 `(1,0) m/s`：A2 率为 `2×exp(−0.5)`；不是 `2×exp(−1)`。
- 单脚一次平坦有效落脚：T1 的单步贡献为 `10×1×0.02=+0.2`；保持接触下一步不重复得到该项。
- 两脚有效且接触全匹配：T2 的单步贡献为 `1.5×0.02=+0.03`；两脚标签全无效则为 0。
- 单脚满足 stumble 力阈值：T4 的单步贡献为 `−20×1×0.02=−0.4`；持续满足则每控制步计罚。
- 单脚一次快速切换：T5 单步贡献 `−30×0.02=−0.6`。
- 单脚力 450 N：按当前无阈值 T6 实现，单步贡献 `−1e−6×450²×0.02=−0.00405`。

现有公式来源：[pgmt_math.py](../pmt_tasks/pgmt_math.py#L140)、[实际权重/尺度](../configs/reward/pgmt_stage1.yaml)、[状态及差分历史](../pmt_tasks/pgmt_runtime.py#L194)。候选接触/恢复公式起点：[本地独立 PGMT](../../PGMT/pgmt/environment.py#L263)；该实现不是论文未公开参数的权威来源。P3 实施时若调整任何门槛、求和/均值、事件年龄、有效点要求或难度来源，应同步修改本节及保存到 checkpoint 的奖励契约。
