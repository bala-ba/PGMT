# PGMT v6 P1 补充验收

2026-10-11：**P1 在 v6、动作固定第 0 帧、随机出生 XY/yaw 的当前范围内验收完成。** 本次实际运行 Isaac Sim 6 / PhysX，补齐随机初始化、扫描与续训恢复，并验证常用训练入口。总报告：[acceptance_20261011.json](../logs/pgmt_v6_acceptance/acceptance_20261011.json)。

## 结果

| 项目 | 实际覆盖与结果 | 独立报告 |
| --- | --- | --- |
| 全库候选几何 | 862 个动作；每级 3,951 个 CSV 允许的动作/样式组合；十级×16 小块×每组合 2 个随机 XY/yaw，共 **1,264,320 个候选**。282,577 个不满足放置/未来窗口边界条件的候选被拒绝；每个动作/样式/等级组合均找到有效候选 | [随机初始化](../logs/pgmt_v6_acceptance/20261011_r2/poses/validation.json) |
| 实际机器人 reset | 每个允许的动作/样式/等级组合调用真实 command reset，共 **39,510 次**。800 小块均有实际随机出生；最大重试 9 次，上限 128；保守碰撞净空最小 **2.990 mm**（3 mm 目标、10 μm 数值容差）。未修改动作或兼容表 | 同上；逐批位置、yaw、小块与重试记录为同目录 `reset_samples.jsonl` |
| 独立 PhysX 重叠检查 | 对每个小块选取一个实际接受的机器人姿态，全身 221 个保守球，共 **176,800 次**查询；地形重叠为 0 | 同上 |
| 统一参考 | 实际 reset 与全身参考位置最大差 **0.03052 mm**；command、当前/未来参考一致；关节位置/速度保持源数据；局部 reset 不修改其他环境变换 | 同上 |
| 原生扫描 | 800 小块×四个 yaw，共 **3,200 个传感器位姿**；2×2 m、21×21、row=y/col=x。含边界检查共 **1,433,250 个扫描单元**，独立 PhysX 探针 **4,171 个**。扫描对 GPU 网格最大差 **6.676 μm**，对 PhysX 最大差 **4.567 μm** | [扫描](../logs/pgmt_v6_acceptance/20261011_r2/scan/validation.json) |
| 扫描边界与刷新 | 倒金字塔负高度保留；跨小块、等级、样式查询不改出生分配；四侧地图边缘各出现 42 个正确无效单元。仅扫描越界不 done，实际机器人越界 truncation，reset 后扫描与当前位置一致；wrapper 保留两个 `[N,21,21]` 张量 | 同上 |
| 实际续训恢复 | 新进程载入 `model_15999.pt`，逐项精确比较策略、优化器、iteration、课程等级/样式/origin/统计及 sampler EMA、待合并计数、更新时钟；真实训练 25 次更新，再保存、扰动运行状态、重新加载并追加 1 次更新，合计 **26 次 PPO / 624 控制步** | [续训](../logs/pgmt_v6_acceptance/20261011_r4/resume/validation.json) |
| 恢复后生命周期 | **209 次 done / 209 次 sampler outcome**；课程新增完成 96、失败 106、地图越界 7。689 次初始化/reset、99,631 次参考不漂移、99,840 次接触索引检查通过。手动/恢复 reset 不计 outcome；不匹配的放置配置在修改状态前被拒绝 | 同上 |
| 常用训练入口 | `train_pgmt_p1_gui.sh --headless` 直接加载同一 v6 checkpoint，成功追加 **2 次更新**，`curriculum_initialized_fresh=False`，保存新结果并正常关闭 | [入口状态](../logs/pgmt_v6_acceptance/20261011_cli/training_status.json) |
| 回归检查 | `tests/test_pgmt*.py` 加新增 normalizer 恢复测试，**119 项通过**；Python / shell 语法及 diff 检查通过 | [测试输出](../logs/pgmt_v6_acceptance/unit_tests.log) |

已核对原 Stage 1 `model_15000.pt`、v6 `model_15999.pt` 和 v5 六个保留文件的哈希，均未改变。所有验收产物写入独立目录；原有 1000 次盲走训练报告保留。

## 本次发现并处理的问题

| 问题 | 处理及复验 |
| --- | --- |
| 常用训练入口要求课程统计为零，已有 v6 checkpoint 会被该断言拒绝 | `scripts/train_pgmt_p1.py` 仅在没有 `environment_state_dict` 的首次初始化时检查零统计；同阶段恢复沿用 runner 的严格身份/状态校验。实际入口追加两次 PPO 成功 |
| PPO rollout 后再次 `load_state_dict` 报 inference tensor 原地更新错误 | `EmpiricalNormalization.update()` 原来用新张量替换 `_std`，使其在 inference mode 下变成只适于该上下文更新的张量。改用 `_std.copy_(sqrt(_var))` 保留已注册缓冲区，统计公式及 checkpoint 键不变；新增回归测试，并重跑完整二次恢复验收通过 |
| 新验收脚本沿用 contact sensor 的调试可视化，触发 Sim 6 旧材质接口错误 | 与既有训练入口一致关闭 `contact_forces.debug_vis`，传感器及接触物理继续工作 |
| 验收脚本在 runner 的 inference rollout 之外直接手动 reset | 按现有 runner 调用约定，在 `torch.inference_mode()` 中执行该仿真 reset；恢复统计前后精确一致 |

失败尝试保留：`20261011_r1/poses`（调试材质）、`20261011_r2/resume`（验收调用上下文）、`20261011_r3/resume`（normalizer 二次加载）。这些报告均为 `passed=false`，未计入最终通过结果。成功日志未出现 PhysX 碰撞栈溢出、丢弃接触或未处理异常。

## 范围

- 随机的是出生 XY/yaw；动作起始帧仍固定为 0，足部有效支撑起始限制保持关闭。
- 全库覆盖离散的兼容动作、样式、等级和小块，并在每组合上检查有限随机候选；没有宣称穷尽所有连续位置/朝向。运行期继续执行几何拒绝与重试。
- 独立 PhysX 重叠检查覆盖每个小块的一组实际姿态；其余实际 reset 使用同一已校验网格的全身距离/高度检查。
- 扫描使用受控 torso 位置和四个朝向，测试轴序、符号、完整地图、掩码及刷新。它不向当前盲走策略注入感知。
- 续训恢复训练状态后，从第 0 帧重新初始化仿真 episode；不承诺逐位恢复中断时的物理轨迹或随机数序列。
- Terrain-Glimpse/IFM、感知噪声、地形接触奖励、第四价值头、恢复机制和最终性能仍属于后续 P2/P3/P4，P1 通过不等同于策略收敛。

## 重跑方式

在 PMT 根目录执行，`new_run` 应使用未有验收报告的新目录。三个模式默认 headless、`cuda:0`，复用与可视化训练相同的 Sim 6 / PMT Lab 组合：

```bash
for check in poses scan resume; do
  bash scripts/validate_pgmt_v6.sh \
    --check "$check" \
    --output "logs/pgmt_v6_acceptance/new_run/$check"
done
```

默认续训源是 `logs/pgmt_v6_blind/20261011_02/model_15999.pt`，保持 160 个环境以匹配保存的课程分配。可用 `--checkpoint` 指定同结构的 v6 checkpoint。

常用训练入口的短续训检查：

```bash
bash scripts/train_pgmt_p1_gui.sh --headless \
  --terrain_asset assets/terrain/pgmt_stage2_v6 \
  --checkpoint logs/pgmt_v6_blind/20261011_02/model_15999.pt \
  --num_envs 160 --max_iterations 2
```

去掉 `--headless` 可显示 GUI。默认输出到新的时间戳目录，不覆盖源 checkpoint。
