# PGMT PhysX 大额缓冲区分配定位

日期：2026-10-13。结论：**原生 Cloner 的环境 ID 包围盒编码，与当前共享的 804 个地形 collider 组合，造成候选碰撞对爆增。** 1024 环境的 27.04 GiB 请求可以精确对应到接触流形缓存。不是由 P2 感知网络产生，也不是 `gpu_collision_stack_size=256 MiB` 本身。

**后续正式接入（2026-10-14，P4）：** [pgmt_physx.py](../pmt_tasks/pgmt_physx.py) 已将同一修复接入 Stage2 正式训练入口，保留环境碰撞过滤，设置值纳入 checkpoint 训练契约。5 环境实测重叠机器人隔离及跨四类 tile 接触通过；1024 环境已完成 3 次完整 PPO 更新。其余扩容/恢复和正式训练的最新状态见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。下文“仅诊断、未接入”的表述保留原诊断阶段范围，不代表当前入口状态。

本次仅增加独立诊断脚本、日志和文档；**没有修改正式训练配置、安装环境源码、地形资产或 checkpoint**。1024 的通过范围为无策略的初始化、reset、3 个控制步，不能替代完整 PPO 扩容验收。

## 原因与代码链

1. `InteractiveScene` 在 `replicate_physics=True, filter_collisions=True` 的 GPU 场景调用 Cloner，启用环境 ID。实际加载的原生代码为：
   - `/home/lzp/miniconda3/envs/pmt/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py`，`clone_environments()` / `replicate_physics()`。
   - `/home/lzp/miniconda3/envs/isaacsim/lib/python3.12/site-packages/isaacsim/exts/isaacsim.core.cloner/isaacsim/core/cloner/impl/cloner.py:183–189`：写入 **`physxScene:envIdInBoundsBitCount = 4`**。诊断在首次物理 reset 前读回确认。
2. PGMT 通过 [PGMTTerrainImporter](../pmt_tasks/pgmt_terrain_importer.py#L35) 将 v6 地形放在共享 `/World/ground`，共有 **800 个小地形 mesh + 4 个边界 mesh**。它们不属于某个机器人环境 ID，需与所有环境碰撞。
3. PhysX GPU broadphase 的 `translateAABBsLaunch()` 在开启 ID 位编码时，会把共享对象的编码包围盒设为覆盖整个空间。因此，每个机器人的碰撞形状都与全部 804 个 mesh 建立候选碰撞对，远处 mesh 也不能在此阶段排除。环境间机器人碰撞过滤是另一段逻辑，不等于关闭这项位编码。
4. 当前运行资产每个机器人有 **28 个 capsule + 1 个 sphere = 29 个碰撞形状**。这些形状与 mesh 的候选对会分配 `PxgPersistentContactMultiManifold`，结构包含 4×6 个接触槽及相对位姿等，16 字节对齐后为 **1216 字节**。`prepareTempContactManagers()` 按 `sizeof(Manifold) * nbNewManagers` 分配。

因此，1024 环境的 mesh 候选对与请求恰好为：

```text
1024 × 29 × 804 = 23,875,584 个候选对
23,875,584 × 1216 = 29,032,710,144 bytes ≈ 27.04 GiB
```

这与原始 OOM 日志的请求字节数**完全一致**。这是仅一项接触缓存，不包含其他 PhysX、RTX、动作库和网络内存。候选对并不表示发生了这么多真实接触。

机制核对使用 NVIDIA 官方 `110.1-omni-and-physx-5.9.0` 固定提交 `517a0073715120e114ee055b63b26c95e00d9039`：

- [GPU broadphase 的共享对象包围盒及环境过滤](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/gpubroadphase/src/CUDA/broadphase.cu#L96)。
- [1216 字节接触流形结构](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/gpunarrowphase/include/PxgPersistentContactManifold.h#L80)。
- [按候选对数分配接触流形](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp#L8202)。同版本 `convexTrimeshMidphase` 报错行 1998 也与本机日志对应。
- [官方属性说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html)：`0` 关闭包围盒 ID 编码并保留更高的空间编码精度；环境 ID 过滤本身独立保留。

## 单变量对照

同一 v6、同一机器人 USD、seed=722、cuda:1、headless、无感知/无策略/无 PPO。只在独立诊断进程中修改指定变量。采样使用原生 `IPhysxStatistics.get_physx_scene_statistics()`；总显存为 `nvidia-smi` 整卡快照，含显示与 RTX，不是 Torch 峰值。

| 配置 | 首次初始化候选对 | 初始化 narrowphase heap | 第 3 控制步总显存 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 160，原配置，bits=4、spacing=0 | 3,730,880 | 6.176 GiB | 16,190 MiB | 完成 |
| 160，仅 spacing=3 m | 3,730,880 | 6.176 GiB | 16,171 MiB | 完成；没有改善 |
| 160，仅 bits=0，spacing 仍为 0 | 4,160 | 1.053 GiB | 10,072 MiB | 完成 |
| 1024，仅 bits=0，spacing 仍为 0 | 26,624 | 1.085 GiB | 11,049 MiB | 完成初始化、reset、3 步 |

160 原配置初始化的 `3,730,880 = 160×29×804 + 320`；mesh 对占绝大多数。bits=0 后候选对按真实空间位置筛选；第三步分别为 2,991（160 环境）和 19,704（1024 环境），地形接触正常产生，1024 第三步记录 15,306 个 rigid contacts。

初始机器人确实全部位于 `(0,0,0.78)`，但将其分散为 160 个位置仍得到完全相同的候选对和窄相内存，**因此重叠初始化不是这次大额分配的主因**。bits=0 对照保留了原重叠初始化、环境 ID 过滤、完整地形以及 PGMT reset 随机出生逻辑。

此前 1024 原配置运行已在 `headless=True, enable_cameras=False, profile=blind` 下失败，故不能归因为打开 GUI 或感知注入。控制实验仍使用原完整 Kit experience，没有声称 RTX/显示完全不占显存。

## 建议的正式修复

在 PGMT 场景克隆/物理复制注册完成后、首次 `sim.reset()` 前，把当前 PhysicsScene 的 `physxScene:envIdInBoundsBitCount` 设置为 **0**，并读回检查。保留 `replicate_physics=True` 与 `filter_collisions=True`。设置过早会被原生 Cloner 再写回 4；设置到训练 reset 阶段则已经错过首次大额分配。

该修改恢复共享地形的空间粗筛，适合 PGMT 跨 tile 查询与接触语义，无需复制整张地图到每个环境。不要用增大 collision stack 或删除碰撞体来解决此问题。正式接入后应补充环境间碰撞隔离/跨 tile 接触验证，以及 1024 环境完整 P2 PPO、保存恢复与吞吐验收；本次没有修改这些正式路径。

## 复现与材料

- [独立诊断脚本](../scripts/diagnose_pgmt_physx_memory.py)：复用独立进程 PhysX 错误监视器，不加载 checkpoint。
- [启动环境脚本](../scripts/diagnose_pgmt_physx_memory.sh) 已从本地诊断记录整理入库。默认路径对应当前机器，可用 `PGMT_LAB_SOURCE`、`PGMT_GUI_ENV` 覆写；机器人 USD 和地形/动作资产仍需按 P2 文档准备。例如在项目根目录：

```bash
bash scripts/diagnose_pgmt_physx_memory.sh \
  --num_envs 1024 --env_id_bits 0 \
  --log_dir logs/pgmt_physx_diagnosis/recheck_1024
```

输出目录必须未使用。省略 `--env_id_bits` 保留 Cloner 原配置；`--spacing 3` 为初始克隆间距对照。诊断中间曾修正两个脚本接入问题：`height` 与 AppLauncher 参数重名；原生 EventManager 不允许物理复制场景使用 prestartup 随机化。因此最终探针在独立进程包装 `InteractiveScene.__init__`，场景创建后直接读属性/做对照，不使用 prestartup 事件。以下日志链接对应本地材料，不随 Git 上传；关键结果及哈希已归档到 [2026-10-14 版本快照](versions/pgmt-p2-2026.10.14.json)。

- [汇总 JSON](../logs/pgmt_physx_diagnosis/report.json)，包含原始状态路径、精确公式和范围。
- [160 原配置](../logs/pgmt_physx_diagnosis/baseline_160_b/training_status.json)、[间距对照](../logs/pgmt_physx_diagnosis/spaced_160/training_status.json)。
- [160 bits=0](../logs/pgmt_physx_diagnosis/bits0_160/training_status.json)、[1024 bits=0](../logs/pgmt_physx_diagnosis/bits0_1024/training_status.json)。
- 原始失败：[1024 OOM 日志](../logs/pgmt_p245/cost_1024_blind.log)，请求为 `29,032,710,144 bytes`。该运行的训练数据仍全部作废。

本次有限诊断均已退出；结束时 `nvidia-smi --query-compute-apps` 没有计算进程。
