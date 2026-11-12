## 直接运行头部深度相机仿真

```bash
cd /home/lzp/projects/perceptive_tracking/PBFM/PMT
bash scripts/pmt.sh mujoco-live
```

通过 tutorial VNC 查看。此入口在窗口打开期间持续执行策略推理、`mj_step` 和头部深度相机采集，查看器与物理循环共用当前 `MjData`，不读取录像或 `states.jsonl`。旁边的面板显示当前高程输入和最新深度图，场景含相机光心、方向与视野标记。

在 MuJoCo 窗口获得键盘焦点后：空格暂停/继续，`R` 重置当前动作，`F` 切换跟随视角，`Q` 退出。也可点击右侧面板顶部红色 **STOP SIMULATION** 按钮，或关闭任一窗口。面板支持 `Esc`/`Ctrl+C`；主查看器可能优先处理 `Esc`，建议用 `Q` 或停止按钮。动作片段结束后自动重置物理、观测历史和局部地图，重新执行策略。鼠标可调整观察视角。

从任何终端停止本项目的实时仿真：

```bash
bash scripts/pmt.sh mujoco-stop
# 若有多份实例，仅停止指定的 PMT 实时仿真 PID：
bash scripts/pmt.sh mujoco-stop --pid 12345
```

停止命令核对用户、脚本路径和进程启动时间，先请求正常退出，无响应时升级为终止/强制结束；不停止 VNC 或训练进程。`Ctrl+C` 只作用于启动终端的前台进程组，无法关闭从其它终端/后台启动的仿真。实时入口显式处理 SIGINT/SIGTERM，GUI 清理超时 6 秒后退出，避免窗口线程一直占住进程。

默认尝试按真实时间运行；服务器软件渲染负载较高时会慢于现实时间，不跳过物理计算。200 步验证执行了 800 次 `mj_step`、101 帧深度采集，无跌倒和仿真警告，墙钟约 19 s、仿真时间 4 s。可用 `--steps 200` 限制运行步数，用 `--depth-pitch` 调整仿真俯角，用 `--motion-index` 选择动作摆放。

输出自动放在 `logs/depth/live/<时间戳>`；退出时保存 `live_result.json`，默认不录制视频。当前发布权重的视觉门控限制仍适用，详见 [DEPTH_PERCEPTION.md](DEPTH_PERCEPTION.md)。

## 可视化训练地形
查看场景
```bash
cd /home/lzp/projects/perceptive_tracking/PBFM/PMT

MJ_PYTHON=/home/lzp/miniconda3/envs/pmt/bin/python \
bash /home/lzp/projects/tutorial/mujoco_gui.sh model \
TCRS/assets/g1/g1_29dof_scene_stairs_ud.xml
```

查看已经录制的仿真轨迹
```bash
cd /home/lzp/projects/perceptive_tracking/PBFM/PMT

bash scripts/pmt.sh replay \
logs/visualization/terrain_teacher --loops 1
```

实时运行教师策略查看
```bash
cd /home/lzp/projects/perceptive_tracking/PBFM/PMT

bash scripts/pmt.sh gui-play \
--task PMT-WalkDanceBigMap-G1-v0 \
--resume_path checkpoints/pretrained/walkdance_bigmap_teacher.pt \
--num_envs 1 --max_steps 3000 --real_time

# --motion_file可以指定某个npz
# --real_time 只是限制运行速度，使仿真尽量按现实时间播放
```

## 高程图可视化

地形显示采用灰色棋盘地面、按高度区分的暖色台块和侧向光照。Bigmap 中的方块逐块绘制，突出顶部、侧面和台阶边界；颜色只是显示辅助，不改变策略高程输入或物理碰撞。2026-09-24 前导出的蓝色地形视频不会自动更新，重新运行 `replay` 或 `gui-play` 可使用新样式。

`gui-play`、`gui-train` 会自动显示任务中的 `height_scanner`：

- 机器人周围的三维网格对应 Isaac Lab 射线实际命中的位置，随躯干朝向移动。
- VNC 中，MuJoCo 旁边的独立高程面板显示两张热图；导出视频将同一面板叠加在左下角。`Terrain z (m)` 是局部地面的世界坐标高度；`Policy scan (m)` 是同一步返回给 student 的相对高度观测（含原有噪声，在网络归一化之前）。
- 两张热图均为机器人前方朝上、左侧朝左；网格为 17 × 11，间距 10 cm。颜色尺固定，图下另显示当前有效值的范围。
- 粉色表示无效／随机丢弃的采样点。显示直接读取已有观测，不重新生成噪声或改变策略输入。
- Teacher 不使用高程输入，右图会标注 `No policy scan (blind teacher)`，左图和三维网格仅用于观察地形。

运行已发布的 PPOft student，同时录制带高程图的视频：

```bash
bash scripts/pmt.sh gui-play \
  --task PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0 \
  --resume_path checkpoints/pretrained/ppoft_vision_bigmap_walkdance.pt \
  --num_envs 1 --max_steps 3000 --real_time \
  --video --video_length 300 --video_dir logs/visualization/my_student_height_scan \
  network.base_policy_ckpt=checkpoints/pretrained/ppoft_vision_bigmap_walkdance.pt
```

最后一项让初始化使用本地发布的权重，避免默认配置引用原始训练服务器上的 checkpoint 路径。

已验证的示例为 `logs/visualization/student_height_scan/testing.mp4`。在 VNC 中回放：

```bash
bash scripts/pmt.sh replay logs/visualization/student_height_scan --loops 10
```

新录制的 `states.jsonl` 包含实际射线命中、策略高程值及有效性掩码，回放会保留同一份高程观测。旧录像未记录这些数据，仍可回放机器人动作，但不会补画高程图。

## 深度摄像头
```bash
bash scripts/pmt.sh mujoco-live

# --motion index 49 指定参考动作，索引范围是0-98
```
## 独立 MuJoCo 物理测试

新增[深度相机感知与 G1 接入](DEPTH_PERCEPTION.md)。相机闭环录像可同时显示机器人、深度图与相机重建高程：

```bash
bash scripts/pmt.sh replay logs/depth/head_camera_motion0 --loops 1
```

当前头部安装对应视频为 `logs/depth/head_camera_motion0_visualization/testing.mp4`。相机 RGB/深度/高程并排视频为 `logs/depth/head_camera_motion0/depth_perception.mp4`。粉色包括视野外、自身遮挡和历史过期的未知区域。旧 `student_motion0` 录像仍为此前的胸部安装。

深度场景回放现在还会标识相机：黄色点为光心、黄色箭头为光轴方向、青色线框为视野角范围，标签为 `Depth camera (sim)`。标记读取录制场景 XML 中的实际安装参数，并随机器人运动；只添加显示图元。根据 `assets/G1_structure.png`，默认位置已移至头部正面上方（`head_link` 局部坐标 `[0.08, 0, 0.48] m`），而非胸部。图示仅用于近似定位，俯角保留为可配置的 55°，并非实测光轴标定。青色线框画到 45 cm 处以展示角度，不表示最大测距。

头部相机预览：`logs/depth/head_camera_motion0_visualization/preview.png`。回放需要保留原录制元数据指向的场景 XML；若缺失，相机标记不会凭空猜测安装位置。查看器模型把固定头部合并在躯干中，因此回放会将相机安装变换到等价的躯干坐标，标注的物理位置仍在头部，并对照录制的光学位姿检查一致性。

注意：审计发现本地发布 PPOft 权重的两处视觉门控全零。因此这些视频验证了深度数据链路与物理运行，不能证明此权重已学会利用深度观测；细节见上面的感知文档。

当前 PPOft student 已另外完成真实 `mj_step` 闭环测试，详见[独立验证结果与复现命令](MUJOCO_STUDENT_VALIDATION.md)。三组 30 秒测试均未跌倒，但存在位置跟踪漂移。

```bash
# 回放真实 MuJoCo 物理测试（带高程图）
bash scripts/pmt.sh replay logs/sim2sim/student_motion0 --loops 3
```

## tensorboard for stage 2
cd /home/lzp/projects/perceptive_tracking/PGMT

CUDA_VISIBLE_DEVICES='' /home/lzp/miniconda3/envs/pmt/bin/tensorboard \
  --logdir runs/campaign_v1/phase2_train_0001_continuous_1790494140201623853/tensorboard \
  --host 0.0.0.0 --port 6006 \
  --reload_interval 30 --load_fast=false