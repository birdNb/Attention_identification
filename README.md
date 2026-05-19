# Gaze Robot · 注视识别工程

> 基于 **MediaPipe Face Mesh + Face Detection** 的实时注视识别工程。
> 面向 **Jetson Orin Nano** 设计（2K · 140° FOV 广角摄像头），同时在 **PC（Linux + NVIDIA GPU）** 上开箱即用。

---

## 1. 这个工程做什么？

实时判断"摄像头前的人是否在注视相机"，并在持续注视 2 秒后触发 `GAZE LOCKED` 标志位，可作为机器人/智能音箱/数字人 **"被唤醒"** 的低成本视觉信号。

判定为"正在注视"必须 **同时满足三个条件**：

| 条件 | 判定依据 | 反映 |
|------|---------|------|
| **脸朝相机** | 头部 yaw（左右转头）+ pitch（抬头低头）都在阈值内 | 用户正脸朝向摄像头 |
| **瞳孔居中** | 左右两眼的瞳孔在眼眶正中（水平 + 垂直） | 眼球真正"看向"相机 |
| **持续 ≥ 2 秒** | 上面两条同时满足 ≥ `GAZE_HOLD_SECONDS` | 排除瞥一眼 / 误触发 |

画面会用颜色独立反馈三个状态，方便调参时一眼分辨"是脸的问题"还是"是眼睛的问题"：

| 元素 | 含义 | 颜色 |
|------|------|------|
| 人脸框 + "Face" 文字 | `face_ok`（脸是否朝相机） | 绿=朝向 / 红=偏向 |
| 眼睛框 + 瞳孔圆点 | `pupil_ok`（瞳孔是否居中） | 绿=居中 / 红=偏离 |
| 顶部状态条 | 综合状态 | 灰=无脸 / 红=未看 / 青=正在累计 / 绿=锁定 |
| 底部居中绿牌 | 持续 2 秒后弹出 | `LOOKING AT CAMERA` |
| 右上角 | 实时性能 | `FPS` / `CPU%` / `GPU%` |
| 底部 debug 行 | 实时阈值数值 | `yaw / pitch / px / py / FPS [ROI]` |

---

## 2. 项目结构

```
Attention_identification/
├── gaze_robot.py        # 主程序：人脸检测 + 瞳孔定位 + 注视判断 + 系统监控
├── test_camera.py       # 相机测试脚本（强制 MJPG，调通 2K@30FPS）
├── requirements.txt     # 依赖列表
├── setup.py             # 支持 pip install -e .，提供命令行入口
├── setup.sh             # 一键 conda 环境 + 依赖安装
└── README.md
```

安装后会注册两个命令行命令：

```bash
gaze-camera-test    # 等价于 python test_camera.py
gaze-robot          # 等价于 python gaze_robot.py
```

---

## 3. 安装

### 方式 A：一键脚本（推荐 Jetson）

```bash
bash setup.sh
```

脚本会：

1. 创建 conda 环境 `gaze_robot`（Python 3.8）
2. 安装 numpy / opencv-python / pillow / mediapipe / onnxruntime / psutil
3. 打印使用提示

### 方式 B：手动安装

```bash
conda create -n gaze_robot python=3.8 -y
conda activate gaze_robot
pip install -r requirements.txt
pip install -e .          # 注册 gaze-robot / gaze-camera-test 命令
```

### 可选：装 pynvml 让 PC GPU 占用显示更准

```bash
pip install nvidia-ml-py
```

Jetson 不需要装这个（会自动用 sysfs 读取 GPU 占用率）。

---

## 4. 使用

### 步骤 1：测试相机能否打开

```bash
conda activate gaze_robot
python test_camera.py
```

正常应看到左上角显示 **`FPS: ≈30`** 和 **`编码=MJPG`**。

> 如果 FPS 只有 1~5，说明相机走了 YUYV，请检查 `USE_MJPG=True` 是否生效。

### 步骤 2：跑注视识别

```bash
python gaze_robot.py
```

启动后：

1. 正对相机 2 秒以上 → 底部弹出绿色 `LOOKING AT CAMERA`
2. 头偏向 / 低头 → 人脸框变红，触发被中断
3. 故意瞟向旁边 → 瞳孔点变红，触发被中断
4. 按 **ESC** 退出

### 步骤 3：根据现场环境调阈值

跑起来后看画面底部那行小字：

```
yaw=+0.02  pitch=0.68  px=0.07  py=0.11  FPS=29.8 [ROI]
```

| 字段 | 含义 | 正眼直视典型范围 |
|------|------|----------------|
| `yaw` | 偏航（左右转头） | ±0.05 内 |
| `pitch` | 俯仰（抬头低头） | 0.55 ~ 0.85 |
| `px` | 瞳孔水平偏离正中 | < 0.10 |
| `py` | 瞳孔垂直偏离基线 0.42 | < 0.15 |
| `FPS` | 实测帧率 | ≈相机帧率 |
| `[ROI]` | 远距离兜底标记 | 跟丢被救回时显示 |

把 `gaze_robot.py` 顶部参数按现场数值微调即可（见下文"调参指南"）。

---

## 5. 技术实现流程

### 5.1 整体流水线

```
                    相机 (2K@30fps · MJPG)
                            │
                            ▼
                  cv2.VideoCapture (V4L2 + MJPG)
                            │  BGR frame
                            ▼
                  cv2.cvtColor → RGB (writeable=False)
                            │
                ┌───────────┴───────────┐
                │ 第一阶段：face_mesh    │
                │ 全图 468 关键点跟踪    │
                └───────────┬───────────┘
                            │
                       检测到？
                       ├── 是 ──> 直接进入判定阶段
                       │
                       └── 否 (远/快/丢)
                            │
                            ▼
                ┌────────────────────────┐
                │ 第二阶段（兜底）        │
                │ face_detection (m=1)   │  ←─ 覆盖 0~5m 远距离
                │ 找最大人脸 bbox        │
                │ pad 35% → ROI          │
                │ face_mesh_roi 二次处理 │  ←─ 等效"放大人脸"
                │ landmarks 映射回全图    │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ 注视判定 evaluate_gaze │
                │  ├─ face_orientation   │
                │  ├─ pupil_centering    │
                │  └─ AND → looking      │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ GazeHoldTimer          │
                │  瞬时 looking 累计 2s  │
                │  → locked = True       │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ 绘图：人脸框 / 眼框 /  │
                │ 瞳孔点 / 状态条 /      │
                │ 大字弹牌 / FPS/CPU/GPU │
                └────────────────────────┘
```

### 5.2 关键算法

#### ① 脸部朝向（`face_orientation`）

用 Face Mesh 的 5 个关键点直接几何估算，**不调用 PnP**，比标准头部姿态估计快 10 倍：

| 名称 | 计算 | 物理意义 |
|------|------|---------|
| `yaw` | `(鼻尖.x - 双眼外角中点.x) / 眼宽` | 左右转头，0 = 正脸 |
| `pitch` | `(鼻尖.y - 前额.y) / (下巴.y - 前额.y)` | 抬头低头，0.7 ≈ 正脸 |

判定：`abs(yaw) < YAW_TOL  AND  PITCH_MIN < pitch < PITCH_MAX`

#### ② 瞳孔居中（`pupil_centering`）

把眼眶看成一个局部坐标系：

```
  外角(33)  ──────────  内角(133)     y方向：上眼睑(159) → 下眼睑(145)
        ←  x 0~1  →
              ▲
              瞳孔(468) 在 0.5 附近 = 水平居中
```

- 左右眼分别计算 `x_ratio` / `y_ratio`，再分别求与基线的偏差
- **取左右眼的最大值**（而不是平均），避免一眼居中一眼斜被均值"洗掉"
- 阈值：`max_x_dev < PUPIL_X_TOL  AND  max_y_dev < PUPIL_Y_TOL`

> 注意：垂直基线设为 `PUPIL_Y_BASE = 0.42` 而不是 `0.5`，因为人眼解剖上瞳孔自然位置就略偏上。

#### ③ 持续 2 秒触发（`GazeHoldTimer`）

简洁状态机：

- `instant_looking = True` 且 `start_t is None` → 开始计时
- 累计 `elapsed ≥ GAZE_HOLD_SECONDS` → 置位 `locked = True`
- 一旦 `instant_looking = False` → 立即清零并重置

这样能严格避免"瞥一眼就触发"。

#### ④ 远距离 / 快速移动跟丢兜底（关键优化）

MediaPipe Face Mesh 内部会把输入压缩到 ~256×256 处理，远距离的小脸压缩后只剩几十像素，检测器找不到。解决：

```
全图 mesh 没结果
        │
        ▼
face_detection (model_selection=1)   ← 专为 0~5m 远距离训练
        │
        ▼ 取置信度最高的 bbox，pad 35%
ROI = rgb[y1:y2, x1:x2]              ← 把远脸"裁剪放大"
        │
        ▼
face_mesh_roi (低 confidence)        ← 独立实例，避免跟踪状态冲突
        │
        ▼
remap_landmarks() 把 ROI 内归一化坐标映射回全图
```

效果：远距离 / 快速移动跟丢一两帧后立即被救回，**不用凑近重新启动**。

性能开销：仅在 mesh 跟丢的那一帧才走兜底，平均约 +5~15ms，30fps 不掉帧。

#### ⑤ 性能监控（`SystemMonitor`）

独立后台线程每 0.5s 采集一次：

- **CPU**：`psutil.cpu_percent`
- **GPU**：自动按优先级选 backend
  1. `pynvml`（PC NVIDIA 最准）
  2. Jetson sysfs（`/sys/devices/.../gpu*/load`，**浅层 glob**，避免 `/sys/devices/` 符号链接死循环）
  3. `nvidia-smi`（备用兜底）
  4. 都失败 → `GPU: N/A`

线程是 daemon，主进程退出自动结束，不卡 ESC。

---

## 6. 调参指南

所有可调参数都集中在 `gaze_robot.py` 顶部：

```python
# 相机
WIDTH, HEIGHT = 2560, 1440      # 改成 1920x1080 跑得更流畅
TARGET_FPS = 30
CAM_ID = 0                      # 改成 1 / 2 如果第一个相机不是目标
USE_MJPG = True                 # USB 高分辨率必须 True

# 脸部朝向（越小越严）
YAW_TOL = 0.13                  # 左右转头容差
PITCH_MIN, PITCH_MAX = 0.45, 0.95

# 瞳孔居中（越小越严）
PUPIL_X_TOL = 0.10
PUPIL_Y_BASE = 0.42             # 垂直基线（看相机时实测 py 数值）
PUPIL_Y_TOL = 0.20

# 触发时间
GAZE_HOLD_SECONDS = 2.0

# 远距离兜底
ENABLE_ROI_FALLBACK = True
DETECT_CONFIDENCE = 0.3         # 越低越能侦测远脸（会更多误检）
ROI_PAD_RATIO = 0.35            # ROI 四周扩展比例
```

### 常见调参场景

| 现象 | 修改方向 |
|------|---------|
| 正眼看相机也不触发 | 调大 `PUPIL_X_TOL` / `PUPIL_Y_TOL`，或修正 `PUPIL_Y_BASE` 为实测 py 中位数 |
| 头稍微歪就触发不了 | 调大 `YAW_TOL` |
| 太容易误触发 | 调小 `PUPIL_X_TOL` 和 `YAW_TOL` |
| 想更快/更慢锁定 | 改 `GAZE_HOLD_SECONDS` |
| 远距离跟踪失败 | `DETECT_CONFIDENCE` 调到 0.2 |
| 跟丢后 ROI 把脸切到边缘 | `ROI_PAD_RATIO` 调到 0.5 |
| FPS 不到 30 | 把 `WIDTH/HEIGHT` 降到 1920x1080 |

---

## 7. CSI 相机适配（可选）

如果你的 Jetson 用 IMX219 / IMX477 等 CSI 相机，把 `open_camera()` 改成 GStreamer 管线：

```python
def open_camera():
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=2560, height=1440, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

---

## 8. 常见问题

**Q: 安装时报 `Could not find a version that satisfies opencv-python==4.8.2.62`**  
A: 4.8.2.62 是无效版本号，`requirements.txt` 已经改成 `4.8.1.78`。如果还报错，放宽到 `opencv-python>=4.8,<4.9`。

**Q: 启动后没画面、Ctrl+C 才能退出**  
A: 旧版 `SystemMonitor` 用了 `glob(..., recursive=True)` 扫 `/sys/devices/`，会卡在符号链接死循环，已修复为浅层 glob。

**Q: 顶部一直显示 `No face`，但人就在画面里**  
A: 远距离/逆光时常见。先确认 `ENABLE_ROI_FALLBACK = True`，再把 `DETECT_CONFIDENCE` 调到 0.2。

**Q: 戴眼镜识别效果如何？**  
A: 戴/不戴眼镜都稳定，因为 `refine_landmarks=True` 用的是 Iris 模型，对反光鲁棒。深色厚框可能让脸朝向判定略漂，可把 `YAW_TOL` 放宽到 0.18。

**Q: 多人怎么办？**  
A: 把 `face_mesh = FaceMesh(max_num_faces=1, ...)` 中的 `1` 改成需要的人数。当前注视判定逻辑只对第一张脸生效，多人触发需自己扩展循环。

---

## 9. 性能参考

| 硬件 | 分辨率 | 帧率 | CPU | GPU |
|------|--------|------|-----|-----|
| Jetson Orin Nano | 1920×1080 | 28~30 FPS | ~35% | ~25% |
| Jetson Orin Nano | 2560×1440 | 18~22 FPS | ~50% | ~40% |
| PC (RTX 4070 Ti Super) | 2048×1536 | 30 FPS（相机上限） | ~10% | <5% |

> 想拉满 Jetson 帧率，建议把分辨率降到 1080p（识别效果几乎没差别）。

---

## 10. License

MIT
