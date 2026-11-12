# PMT / PGMT 常用命令

当前版本：[`pgmt-p4-2026.10.19`](../CHANGELOG.md)。本版记录 collection/PPO 优化、多 GPU、中央 2×2 出生、超时边界容差，以及 Stage1/Stage2 回放。T1 现为 `support_pattern_mean_v2`：双脚实际支撑与参考整体匹配时，持续结算参考支撑脚的平均质量。`train_01` 已按旧 T1 完成 20,000 次更新；`train_02` 按新 T1 训练，归档时仍在运行。两者不可直接按同一契约严格续训，具体状态和权重哈希见 [版本快照](versions/pgmt-p4-2026.10.19.json)。recovery 暂缓，固定评测和收敛验证未完成。以下路径对应当前机器。

Stage2 当前默认地形为 **v8：直接转换 P-BFM 原生训练地形**，包含七类 × 十级的 70 个 5×5 m 地块、84 个保护地块和外围安全带/围墙。初始化采用 **地块中心 XY + 随机朝向（−π～π）**。动作采用 **EMD 权重选择 clip，再在片段内均匀随机选起点**，详见 [EMD 采样与恢复说明](PGMT_EMD_SAMPLING.zh-CN.md)。完整地形参数、训练比例、构建和查看命令见 [v8 地形说明](PGMT_TERRAIN_V8.zh-CN.md)。下文标注 v6 的预览和历史训练记录仍指原资产；旧策略回放按 checkpoint 记录的地形版本加载。

## Stage 1

可视化：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT

DISPLAY=:1 bash scripts/play_pgmt_stage1_flat.sh \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz \
  --speed 1 \
  --log_dir "logs/pgmt_stage1_flat_policy/$(date +%Y%m%d_%H%M%S)"
```

## 地形可视化

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
OMNI_KIT_ACCEPT_EULA=YES \
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/view_pgmt_terrain.py \
  --asset assets/terrain/pgmt_stage2_v5 \
  --gpu 1 --family up_stairs --level 5 --view tile
```

## 支撑 / 摆动可视化

```bash
cd /home/lzp/projects/perceptive_tracking/PMT

MUJOCO_GL=glfw \
/home/lzp/miniconda3/envs/pmt/bin/python scripts/view_pgmt_motion.py \
  --motion /home/lzp/projects/perceptive_tracking/PMT/assets/motions/pgmt_clipped_50hz_tcrs_v1/walk3_subject3_clip22.npz
```

空格暂停/继续，左右方向键逐帧，`R` 回到开头。

## 机器人初始化可视化

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python scripts/inspect_pgmt_placement.py \
  --headless --validate --export-scene --device cuda:0
/home/lzp/miniconda3/envs/isaacsim/bin/python scripts/view_pgmt_placement.py \
  --snapshot logs/pgmt_placement --gpu 1 --family up_stairs --level 5 --yaw-deg 90
```

## P1 无感知地形训练

```bash
cd /home/lzp/projects/perceptive_tracking/PMT

bash scripts/train_pgmt_p1_gui.sh \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --device cuda:1 \
  --num_envs 140 \
  --max_iterations 1000
```

默认跟随单个机器人；在命令末尾追加 `--viewer_mode global` 可进入全局自由视角。训练窗口面板支持随时切换全局、跟随和固定地块视角。

与本次验收配置一致：**开启 Isaac GUI、无感知注入、固定第 0 帧、动作级自适应采样**。从指定权重追加 1000 次 PPO 更新，日志自动新建在 `logs/pgmt_p1_blind/` 下，无需手动激活 Conda。该命令启动新的训练；归档运行 `acceptance_gui_20261010_r14` 已在 390 次更新后由用户停止。

查看固定的 `up_stairs / L5` 地块，并设置相机相对地块 origin 的世界坐标偏移：

```bash
bash scripts/train_pgmt_p1_gui.sh --num_envs 140 --max_iterations 1000 \
  --viewer_mode tile --viewer_terrain up_stairs --viewer_level 5 \
  --viewer_tile_offset 20 -28 48
```

`View tile` 保留全部机器人可见，固定视角不随 reset 或课程升降级移动；选择 L5 只改变查看位置。面板的 `Camera offset (m)` 可调整偏移，点击 `View tile` 应用。详见 [视口模式说明](PGMT_P1_ACCEPTANCE.zh-CN.md#视口模式切换)。

## v6 重复小块地形与盲走训练

v6 包含 `flat / stairs / slope / boxes / rough` 五类、十级、800 个 6×6 m 小块。区域内随机选择小块、XY 和 yaw，动作起始帧固定为 0。详细参数见 [v6 地形说明](PGMT_TERRAIN_V6_PREVIEW.zh-CN.md)。

当前机器的 v6 资产已生成。新环境准备好 v5 资产和 Isaac Lab 后，可在 v6 输出目录尚无 manifest 时构建；构建器拒绝覆盖完整资产：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/build_pgmt_patch_terrain.py --config configs/terrain/pgmt_stage2_v6.yaml --headless --device cpu
```

查看地形：

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/isaacsim/bin/python \
  scripts/view_pgmt_patch_terrain.py --asset assets/terrain/pgmt_stage2_v6 --family stairs --level 5 --view region
```

启动新的 v6 无感知训练：

```bash
bash scripts/train_pgmt_p1_gui.sh \
  --terrain_asset assets/terrain/pgmt_stage2_v6 \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --num_envs 160 --max_iterations 1000 \
  --viewer_mode tile --viewer_terrain stairs --viewer_level 0 \
  --viewer_tile_offset 14 -18 20
```

五类样式的 GUI 名称以 v6 列表为准，使用 `slope / rough`。已有运行 `logs/pgmt_v6_blind/20261011_02` 已完成 1000 次更新并保存 `model_15999.pt`；本命令创建新日志目录。运行结果及碰撞栈修正见 [v6 盲走测试](PGMT_V6_BLIND_TEST.zh-CN.md)。

## 回归测试与 P1 独立验收

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q tests/test_pgmt*.py tests/test_normalization_resume.py -k 'not test_pgmt_four_head'
OMNI_KIT_ACCEPT_EULA=YES /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/accept_pgmt_p1.py --device cuda:0
```

前一条是该环境适用的单元/回归测试；四头真实 checkpoint 专项和完整版本检查见文末“当前版本回归检查”。新增小块布局测试需要本地 `assets/terrain/pgmt_stage2_v6/manifest.json`，其他资产依赖沿用原测试要求。后一条启动 Isaac 完成原 v5 的五组 P1 检查，不代表 v6 专项验收。原高度扫描检查单独运行 `scripts/validate_pgmt_height_scan.py --headless --device cuda:0`。验证范围和历史结果见 [P1 验收记录](PGMT_P1_ACCEPTANCE.zh-CN.md)。


## P2 感知训练与可视化

从 Stage 1 权重初始化新的感知训练，计划执行 1000 次 PPO 更新，每 50 次保存：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT

bash scripts/train_pgmt_stage2.sh \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --num_envs 160 \
  --max_iterations 1000 \
  --save_interval 50
```

默认使用 `cuda:1`、v8 地形、GUI，不需要添加 `--headless`。窗口可选择地形机器人、跟随/全局/固定地块视角，并显示选中机器人的高度图、四个局部块和采样中心。`--initialize-from` 只迁移兼容的旧网络张量；感知层、Adam、采样器和课程重新初始化，更新次数从 0 开始。

先做包含几何、跨地块及地图边界检查的 160 环境、25 次更新验收：

```bash
bash scripts/train_pgmt_stage2.sh --acceptance
```

从已保存的 P2 checkpoint 完整续训，下面示例追加 5 次更新：

```bash
bash scripts/train_pgmt_stage2.sh \
  --resume logs/pgmt_p245/gui_160/final.pt \
  --headless --num_envs 160 --max_iterations 5
```

恢复时核对训练/观测契约，并恢复模型、归一化器、优化器、学习率、课程、sampler 与 RNG；物理 episode 重新开始。建议使用 `final.pt`，checkpoint 内 `next_iteration` 才是已完成更新数；例如原生文件名 `model_24.pt` 对应完成 25 次更新。每次启动使用新的日志目录，默认位于 `logs/pgmt_p245/`。

`bash scripts/validate_pgmt_stage2.sh` 编排完整验收流程，会启动多次仿真；专项命令与已有结果见 [P2.4/P2.5 验收](PGMT_P245_ACCEPTANCE.zh-CN.md)。原 P2 验收范围为 160 环境；PhysX 修复现已接入正式入口，后续四头任务的 1024 PPO/恢复和 8192 训练记录见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)。短程验收通过不代表策略已收敛。脚本中的 Python/Isaac Lab/机器人 USD 路径按当前机器配置，迁移机器时需调整。

## PhysX 显存独立诊断

以下命令在独立进程设置 `envIdInBoundsBitCount=0`，检查 1024 环境初始化、reset 和 3 个控制步，不加载策略或执行 PPO：

```bash
bash scripts/diagnose_pgmt_physx_memory.sh \
  --num_envs 1024 --env_id_bits 0 --steps 3 \
  --log_dir logs/pgmt_physx_diagnosis/recheck_1024
```

默认 `cuda:1`、headless；输出目录须未使用。省略 `--env_id_bits` 保留 Cloner 原设置，可用 `--num_envs 160 --spacing 3` 做间距对照。环境路径可通过 `PGMT_LAB_SOURCE`、`PGMT_GUI_ENV` 覆写。已有结果、原始日志范围及后续训练验收要求见 [显存诊断](PGMT_PHYSX_MEMORY_DIAGNOSIS.zh-CN.md)；P3 的已讨论公式与待实现项见 [奖励审计](PGMT_P3_REWARD_AUDIT.zh-CN.md)。

## P3 接触与离线标签可视化

```bash
cd /home/lzp/projects/perceptive_tracking/PMT

bash scripts/view_pgmt_contacts.sh \
  --log_dir "logs/pgmt_p3_contact_viewer/$(date +%Y%m%d_%H%M%S)" \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz \
  --checkpoint logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --speed 0.5
```

默认平地 L0、GPU `cuda:1`，显示实际承重、接触事件和离线支撑标签。添加 `--paused` 可启动后先暂停，再通过面板单步检查。

- 机器人脚部：绿色＝实际承重，灰色＝未承重。
- 旁侧参考骨架：青色＝离线支撑，蓝色＝离线摆动。
- 黄色高亮：发生 touchdown；高亮持续 0.3 秒，底层事件仍只有一步。

展开右侧 `PGMT P3 | Contact / offline labels` 面板，可查看左右脚受力、计时和事件，并使用暂停、单步、慢放。

## 正式四头地形训练与续训

从指定 Stage1 权重初始化新的四头任务：

```bash
bash scripts/train_pgmt_terrain.sh --headless --device cuda:0 --num_envs 160 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --max_iterations 1000 --save_interval 50
```

支持 Stage1/P2 三头显式迁移，保留 upper/lower/auxiliary 行，新 terrain 行置零；新建优化器、课程和采样状态。三头 checkpoint 不能作为四头的同配置 `--resume`。

当前版本从新 T1 的 `train_02` 已保存检查点续训时，需保持原来的 GPU、环境数、种子和训练配置，并使用新的日志目录。以下检查点内部 `next_iteration=13201`；文件名中的 13200 是零起始更新编号。命令追加 100 次更新，不会停止原训练进程：

```bash
bash scripts/train_pgmt_stage2_native.sh --device cuda:1 --num_envs 8192 \
  --resume logs/pgmt_stage2_center2x2/train_02/model_13200.pt \
  --max_iterations 100 --save_interval 50 \
  --log_dir "logs/pgmt_stage2_center2x2/resume_$(date +%Y%m%d_%H%M%S)"
```

`--env_id_bits` 默认 0，修复共享地形显存问题；该设置、环境数量、奖励/终止及感知配置进入严格训练契约。383/393 更新旧权重还涉及 timeout 和出生配置差异；`train_01` 的 20,000 更新权重缺少新 T1 的 settlement 契约，均不能直接用于当前配置的 `--resume`。旧 policy 仍可推理回放。历史记录见 [P4 执行记录](PGMT_P4_EXECUTION.zh-CN.md)，新奖励规则见 [T1 说明](PGMT_TOUCHDOWN_QUALITY.zh-CN.md)。

## 当前版本回归检查

本次归档保留正在运行的训练，隐藏 GPU 后执行 CPU 回归；其中分布式测试实际运行两进程 Gloo。离线动作工具使用 `pmt` 环境，历史 NumPy 2 checkpoint 的四头专项使用 `isaacsim` 环境：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/home/lzp/miniconda3/envs/pmt/bin/python -m pytest -q \
  tests/test_pgmt*.py tests/test_ppo_logging.py tests/test_normalization_resume.py \
  tests/test_all_tasks_resolve.py tests/test_builder_slice.py \
  tests/test_compat_matrix.py tests/test_group_yaml_valid.py \
  -k 'not test_sonic_agent_cfg_modes_resolve_release_onnx and not test_pgmt_four_head'

CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/home/lzp/miniconda3/envs/isaacsim/bin/python \
  -m pytest -q tests/test_pgmt_four_head.py
```

2026-10-19 结果：第一组 **1012 passed / 85 skipped / 28 deselected**，第二组 **25 passed / 2 skipped**；合计 **1037 通过、87 跳过、0 失败**。跳过项为 CUDA/NCCL 用例；第一组排除的 27 项四头用例由第二组覆盖，另 1 项既有 SONIC 用例因 Isaac 许可交互未执行。拆分原因：NumPy 1 不能读取历史 checkpoint 的 NumPy 2 RNG；`isaacsim` 环境的 SciPy 对两个离线动作测试的 Euler 输入形状行为不同，第三方 pytest 插件也存在依赖冲突。本次未重跑 Isaac 仿真，已有 GPU/仿真验收与本次 CPU 回归分别记录。


## 训练 Stage2

正式 headless 训练使用 `pmt / Isaac Sim 5.1`，GUI 使用 Sim 6 兼容入口。两个脚本均已设置相应环境路径，无需手动拼接 `PYTHONPATH`。Stage2 默认从 v8 的当前 5×5 m 地块中心出生，每次重置随机朝向。

### GUI 训练检查

160 环境，打开 Isaac 窗口；本机默认使用显示 GPU 1：

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
HEADLESS=0 LIVESTREAM=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
bash scripts/train_pgmt_terrain.sh \
  --num_envs 160 \
  --max_iterations 1000 \
  --save_interval 50 \
  --performance all \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir "logs/pgmt_stage2_center2x2/gui_$(date +%Y%m%d_%H%M%S)"
```

### 单 GPU 新训

8192 环境，从 Stage1 初始化当前新 T1 任务，使用新的日志目录：

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --device cuda:1 \
  --num_envs 8192 \
  --max_iterations 20000 \
  --save_interval 500 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir "logs/pgmt_stage2_center2x2/train_03"
```

### 同配置续训

使用新 T1 训练已保存的检查点；新任务完成后也可使用其同配置 `final.pt`：

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --device cuda:1 \
  --num_envs 8192 \
  --max_iterations 20000 \
  --save_interval 100 \
  --resume logs/pgmt_stage2_center2x2/train_02/model_13200.pt \
  --log_dir "logs/pgmt_stage2_center2x2/resume_$(date +%Y%m%d_%H%M%S)"
```

`--max_iterations` 表示本次追加的更新次数。续训恢复模型、优化器、学习率、课程和动作采样状态；GPU 拓扑、环境数量、种子及训练配置须与 checkpoint 一致。此前 383/393 次更新的旧权重，以及采用旧 T1 的 `train_01`，均不能直接代入当前配置续训命令。

### 双 GPU 新训

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --gpus 0,1 \
  --num_envs 8192 \
  --max_iterations 20000 \
  --save_interval 100 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --log_dir "logs/pgmt_stage2_center2x2/train_2gpu_$(date +%Y%m%d_%H%M%S)"
```

`--num_envs 8192` 是总环境数，两张卡各运行 4096。单卡继续使用 `--device cuda:1`，不传 `--gpus`。完整参数、NCCL 覆写与停止/恢复规则见 [多 GPU 训练](PGMT_MULTI_GPU.zh-CN.md)。

### TensorBoard

```bash
/home/lzp/miniconda3/envs/pmt/bin/tensorboard \
  --logdir logs/pgmt_stage2_center2x2 --port 6006
```

访问 http://localhost:6006。每次启动使用新的 `--log_dir`；新训若因显存不足调整环境数，之后续训须保持该数量。

## Stage2 策略回放

下例回放已完成的旧 T1 `train_01` 策略；推理加载允许该历史权重，不代表它能按当前奖励契约严格续训。

```bash
cd /home/lzp/projects/perceptive_tracking/PMT
DISPLAY=:1 bash scripts/play_pgmt_stage2_stairs.sh \
  --checkpoint logs/pgmt_stage2_center2x2/train_03/model_17500.pt \
  --motion assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz \
  --terrain slope \
  --level 1 \
  --direction up \
  --patch_id 5 \
  --speed 1 \
  --max_episodes 0 \
  --log_dir "logs/pgmt_terrain_policy/$(date +%Y%m%d_%H%M%S)"
```

五种地形、难度切换、参考机器人及指定动作规则见 [策略测试](PGMT_TERRAIN_POLICY_TEST.zh-CN.md)；Stage1 回放见 [平地可视化](PGMT_STAGE1_FLAT_PLAY.zh-CN.md)。
