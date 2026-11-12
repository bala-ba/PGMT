# G1 深度相机感知接入

2026-09-24：新增可运行的 MuJoCo 深度相机闭环、通用深度建图模块、G1 自带相机的 ROS 2 接入节点，以及深度图/高程图同步可视化。

**验证范围：仿真与离线数据链路已验证；服务器未连接 G1，未安装 ROS 2，实机 ROS 节点尚未联调。当前发布的 PPOft 权重感知门控为零，不能把成功运行解释为策略已经利用深度感知。**

## 数据链路

```text
校正后的深度图 + 内参 + 同时刻光学相机位姿
  -> 反投影点云、机器人自身点过滤、表面法向/深度断层过滤
  -> 6×6 m 滚动局部高程图（5 cm 网格，历史默认保留 2 s）
  -> 当前躯干中心、仅随 yaw 旋转的 17×11 采样网格
  -> [187 个相对高度, 187 个有效性标记]
  -> 原有 student 高程编码器与动作网络
```

`pmt_tasks/depth_perception.py` 不导入仿真器、不读取地形 mesh，不调用地形射线。输入是沿相机光轴的深度 Z，单位米；光学轴为右/下/前。位姿是 optical camera -> 重力对齐 odom 的 4×4 变换。

策略相对高度为 `torso_z - terrain_z`，裁剪到 [-3, 3]，无效高度为 -3、mask=0。沿用训练时 XY 排列（x 变化最快）。网络内仍执行原来的归一化，不在接入模块重复归一化。

地图保存世界坐标下的历史观测，以应对脚下遮挡；机器人移动时滚动窗口，不循环回填旧数据。未知/过期区域保持无效，不填平地，也不自动补洞。单帧过滤竖直墙面和明显深度断层，每格取上层实测点簇并融合相近高度。此实现针对静态、单层地形，没有 SLAM、位姿协方差传播、可见性清理或多层悬空障碍处理。

## MuJoCo 复现

直接看正在运行的策略、物理和深度感知：

```bash
bash scripts/pmt.sh mujoco-live
```

此入口打开实时 MuJoCo 窗口和相机/高程面板，持续执行新的策略推理及物理步进；空格暂停，`R` 重置，`F` 切换跟随，`Q` 或右侧红色 STOP 按钮退出。也可从其它终端运行 `bash scripts/pmt.sh mujoco-stop`。动作结束后自动开始新一轮仿真。速度受服务器渲染负载限制。下面的 `mujoco-eval`/`replay` 则用于离线评测与记录回放。

```bash
# 在 PMT 根目录运行；输出目录必须是新目录
bash scripts/pmt.sh mujoco-eval --perception depth \
  --motion-index 22 --steps 1499 --record-depth \
  --output logs/depth/my_run22_fixed

# tutorial VNC：机器人场景 + 高程图 + 同步深度图，亦可导出场景视频
bash scripts/pmt.sh replay logs/depth/my_run22_fixed \
  --loops 1 --video --output logs/depth/my_run_visualization
```

默认仿真相机为 160×120、垂直视场 70°、俯视 55°、25 Hz。根据用户提供的 `assets/G1_structure.png`，相机已从胸部移到**头部正面上方**，固定到 `head_link`，光心为该 link 坐标中的 `[0.08, 0, 0.48] m`。该模型的头部网格本身有坐标偏移，不能把 `head_link` 的原点当作头部几何中心；经 URDF 固定关节转换，光心相对 `torso_link` 为 `[0.0839635, 0, 0.436] m`。

**位置依据结构示意图和头部网格近似确定，并非 G1 原装相机的实测外参；结构图不提供光轴角度，55° 俯角保留为仿真参数。** 可用 `--depth-width/--depth-height/--depth-fovy/--depth-pitch/--depth-hz/--map-max-age` 修改；模拟相机帧率须整除 50 Hz 控制频率。新位置对应 `logs/depth/head_camera_motion0`，旧 `student_motion0` 及其录像保留原胸部安装，未将旧深度图冒充为头部相机观测。

使用 MuJoCo 深度缓冲区，保留机器人自身遮挡；分割图只用于移除机器人像素，不用于提取地形高度或透视遮挡物。图像边界扩张一像素以去除自身轮廓混合点。仿真位姿来自 MuJoCo 真值；光学噪声、运动模糊、时间延迟和里程计误差尚未模拟。`raycast` 仍是原评测器默认模式；Isaac 的原训练管线没有自动切换成相机训练。

输出：

- `depth_perception.mp4`：相机 RGB、深度和地图面板；RGB 仅用于显示。
- `depth_frames.npz`：自身点已移除的深度和对应位姿，可独立重建。
- `states.jsonl`：闭环物理状态和实际提供给策略的高程观测。
- `metrics.json`：运行状态、覆盖率、处理耗时。

## G1 自带相机：ROS 2 接入

宇树的[官方设备清单](https://github.com/unitreerobotics/xr_teleoperate/blob/main/Device_zh-CN.md)标注 G1 内置 RealSense D435i。实际设备以驱动枚举为准。使用 [RealSense 官方 ROS 2 驱动](https://github.com/realsenseai/realsense-ros)，在连接相机的 G1 计算单元运行，而不是在没有相机的训练服务器启动设备。

ROS 环境需要 `rclpy`、`sensor_msgs`、`std_msgs`、`std_srvs`、`tf2_ros`，以及 NumPy/SciPy/OmegaConf（PMT 包初始化会导入配置模块）。使用与 ROS 发行版匹配的系统 Python，不直接使用服务器的 Isaac Python 3.11 conda 环境。

相机驱动的基础启动命令：

```bash
ros2 launch realsense2_camera rs_launch.py enable_depth:=true enable_color:=false
```

选择设备支持的深度分辨率/帧率，例如 640×480、30 Hz；参数名依已安装的官方驱动版本核对。不要启用补洞后再把补出的区域当成真实有效观测。

接入前 TF 必须完整且时间同步：

- `odom -> torso_link`：机器人状态估计，odom 的 Z 轴与重力对齐。
- 机器人各碰撞 link 的 TF：由同一时钟的关节状态和 robot_state_publisher 提供。
- 机器人到相机的**实测安装外参**，以及驱动发布的 depth optical frame。代码不硬编码原装相机的安装角度。

启动感知节点：

```bash
# 在已 source ROS 2 和机器人工作空间的终端中，从 PMT 根目录运行
python3 scripts/g1_depth_ros2.py --ros-args \
  -p depth_topic:=/camera/camera/depth/image_rect_raw \
  -p info_topic:=/camera/camera/depth/camera_info \
  -p odom_frame:=odom -p torso_frame:=torso_link \
  -p pixel_stride:=4

ros2 topic echo /pmt/depth_observation --once
```

话题名通过参数覆盖。节点从 `CameraInfo.P` 读取校正后的深度内参，`16UC1` 按 ROS 深度约定的毫米转米，`32FC1` 按米读取，支持行填充和大小端；不是把任意原始设备 Z16 都按毫米解析。每隔 4 像素采样时会同步调整内参。机器人自身点根据随附 G1 URDF 的碰撞体与同一深度时间戳下的 TF 去除，默认外扩 4 cm。碰撞体是近似模型，实机需检查四肢/躯干是否充分覆盖，以及外扩是否过大。

输出 `/pmt/depth_observation` 是 `std_msgs/String` 的 JSON，默认 50 Hz，包含 `timestamp`、`pose_timestamp`、`depth_timestamp`、`sensor_fresh`、`vision` 和 `height_scan`。`vision` 长度 374，可按原有 student 的张量接口接入。

节点只发布观测，不控制电机。相机或躯干 TF 超过默认 0.2 s 的新鲜度阈值时，会拒收/停止发布，或将相机失联后的观测全部置为无效；控制端仍需检查消息时效。odom/时钟发生重置后清空地图：

```bash
ros2 service call /pmt_g1_depth/reset_map std_srvs/srv/Empty '{}'
```

原装相机的真实视野可能与演示相机不同；直接更换相机位置后，覆盖率和缺测分布需要重新评估。此节点不是完整实机状态估计器或机器人控制器。

## 通用深度数据回放

```bash
bash scripts/pmt.sh depth-replay logs/depth/student_motion0/depth_frames.npz \
  --output logs/depth/reconstructed.jsonl
```

NPZ 必需字段：

| 字段 | 形状/含义 |
|---|---|
| `depth_m` | `[N,H,W]`，校正后的光轴深度米值；自身/缺失像素为 0 |
| `intrinsics` | `[width,height,fx,fy,cx,cy]`，固定内参 |
| `world_from_camera` | `[N,4,4]`，对应曝光时间的光学相机到 odom 变换 |
| `timestamps` | `[N]`，秒，非递减 |
| `torso_position` | `[N,3]`，同一 odom 坐标系 |
| `torso_yaw` | `[N]`，弧度 |

同一组 API 可直接用于其它驱动：`LocalElevationMap.integrate(...)` 更新地图，`sample(...)` 在控制时刻生成观测。实时调用需自行串行化；ROS 节点使用单线程 executor。

## 已完成的验证与限制

### 22 号动作的分割图索引越界修复

针对 `IndexError: index 2624 is out of bounds for axis 0 with size 2565`，相机不再调用 MuJoCo 3.3.5 内置的无边界检查分割 ID 解码。现在通过公开渲染接口读取 ID 颜色，按实际场景 ID 映射；未知或越界像素标为无效，不裁剪成另一个物体的 ID。相机自己的离屏缓冲关闭多重采样抗锯齿，避免离散 ID 在物体边缘混合；主查看器设置不变。最终的模型几何 ID 索引也做范围检查。

旧失败记录在第 251 步结束；相同物理轨迹在本次环境未稳定复现原异常，因此另加了包括 ID=2624、稀疏 ID、背景和极大错误 ID 的回归测试。修复后 21 项相关测试通过，`logs/depth/motion22_fixed` 完成 1499 步／29.98 s／750 帧深度采集，未知分割像素计数为 0，无跌倒或仿真警告。`logs/depth/live22_fixed` 在 VNC/Mesa 下完成 300 步／6 s，跨过原报错时刻后正常退出。

直接运行 22 号动作：`bash scripts/pmt.sh mujoco-live --motion-index 22`。离线采集仍用 `mujoco-eval --perception depth --motion-index 22 --steps 1499 --record-depth --output <新目录>`。这些是运行验证，不表示此权重的视觉门控问题已经解决或楼梯跟踪精度已经达标。

### 头部安装与原型记录

头部安装更新后，`logs/depth/head_camera_motion0` 重新完成 300 步／6 s 闭环，输出 151 帧新深度图，无跌倒、重置或数值警告；20 组观测接口一致性检查通过。深度记录离线重建与在线策略输入完全一致。更新后的场景录像为 `logs/depth/head_camera_motion0_visualization/testing.mp4`，300 帧均含头部相机标识，标识与记录光学位姿的最大矩阵误差约 3.6e-15。

MuJoCo 3.3.5 在合并带相机的固定头部节点时发生编译崩溃，深度场景因此保留固定 link（`fusestatic=false`），仍使用原有固定关节变换与刚体惯量。接口、FK 和闭环检查均通过。

以下 30 秒数据属于修改头部位置前的胸部原型记录：`logs/depth/student_motion0` 使用原发布权重完成 1499 个控制步、5996 次 `mj_step`、29.98 s 仿真，750 帧相机图像。无跌倒、无重置、无 MuJoCo 警告；平均网格有效率 79.37%，最低 39.04%。冷启动盲区没有预填地形真值。

离线重新读取相机 NPZ 后，749 个可对齐控制时刻的策略高程和 mask 与在线记录完全一致。仅离线诊断调用地形真值射线：111252 个有效样点的平均高度误差约 3.87 mm，但最大误差为 40 cm，约 1.98% 的样点误差超过 5 cm；台阶边缘存在网格跨越/历史融合误差，不能仅凭平均误差认定台阶边界准确。原型相机理想深度下的误差也不代表 D435i 实测精度。

含相机视频录制的 30 s rollout 墙钟约 87 s；相机渲染、掩码和建图平均约 33 ms/相机帧。没有把此结果描述为实时实机性能。场景回放视频共 1499 帧，50 fps，机器人 FK 与记录完全一致。

验证命令：

```bash
MUJOCO_GL=egl bash scripts/pmt.sh test -q \
  tests/test_depth_perception.py tests/test_mujoco_depth_camera.py \
  tests/test_realsense_perception.py tests/test_height_scan_visualization.py \
  tests/test_runtime_options.py

PYTHONPATH=. /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/check_depth_recording.py logs/depth/student_motion0
```

20 项测试通过，覆盖平面/台阶反投影、yaw/网格顺序、遮挡与过期、不回绕滚动窗口、实际深度渲染、无地形射线依赖、ROS 深度单位/字节序/行步长、采样内参和 URDF 自身点过滤。ROS 2 节点本身尚未在 ROS 环境中集成测试。

### 发布权重的感知门控问题

`ppoft_vision_bigmap_walkdance.pt` 的 `intent_gate.alpha` 和 `residual_gate.alpha` 均全零。这两条门控把高程编码与动作输出隔开，因此本次 rollout 的动作与原射线版本相同。相机观测确实进入编码器，但不能据此证明策略会根据相机调整动作。

在 20 组观测中替换高程输入，视觉 latent 最大变化为 3.79，而动作最大变化为 0 rad。可复现审计：

```bash
PYTHONPATH=. /home/lzp/miniconda3/envs/pmt/bin/python \
  scripts/audit_vision_sensitivity.py --output logs/depth/vision_sensitivity.json
```

需要使用感知分支已有效训练的权重，或重新蒸馏/训练后验证动作对地形输入的响应。没有为制造响应而手动改写发布权重的门控。

## 参考来源

- [Elevation Mapping for Locomotion and Navigation using GPU / 官方代码](https://github.com/leggedrobotics/elevation_mapping_cupy)：参考机器人局部滚动地图、有效性与时间层的设计。相关[地图实现](https://github.com/leggedrobotics/elevation_mapping_cupy/blob/main/elevation_mapping_cupy/script/elevation_mapping_cupy/elevation_mapping.py)。本项目是独立的简化 NumPy 实现，没有声称复现该库的概率融合、漂移补偿和可见性清理能力。
- [PMT 论文附录 A.8](https://arxiv.org/html/2606.08059v1#A8)：深度到高程图部署接口。
- [MuJoCo Renderer](https://mujoco.readthedocs.io/en/stable/python.html)：实际深度/分割渲染；当前环境版本 3.3.5。
- [RealSense 官方 ROS 2 驱动](https://github.com/realsenseai/realsense-ros)：深度话题、内参和光学坐标约定。
