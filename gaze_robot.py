# ==============================================
# Jetson Orin Nano 完整注视识别工程
# 功能：人脸检测 + 瞳孔定位 + 多条件注视判断 + 持续 2 秒触发
# 注视判定 = 脸朝相机(yaw/pitch都正) AND 双眼瞳孔水平+垂直都居中
# 连续满足 GAZE_HOLD_SECONDS 秒后置位 gaze_locked 标志
# ==============================================

import glob
import os
import subprocess
import threading
import time

import cv2
import mediapipe as mp

# ===================== 相机配置 =====================
WIDTH, HEIGHT = 2560, 1440
TARGET_FPS = 30
SHOW_SIZE = (1280, 720)
CAM_ID = 0
USE_MJPG = True
# ===================== 注视参数 =====================
# 脸部朝向：偏航 (nose 相对眼睛中线的水平偏移 / 眼宽)
YAW_TOL = 0.13
# 脸部朝向：俯仰 (nose 在脸高中的相对位置，正脸约 0.55~0.85)
PITCH_MIN = 0.45
PITCH_MAX = 0.95
# 瞳孔水平居中：瞳孔 x 在眼眶 [外角,内角] 中归一化偏离 0.5 的容差
# 0.15 ≈ 瞳孔在眼睛水平正中 15% 眼宽范围内即视为居中
PUPIL_X_TOL = 0.10
# 瞳孔垂直居中：瞳孔 y 在眼眶 [上,下] 中归一化偏离基线的容差
# 注：生理上瞳孔自然位置约在眼眶 0.42 处（略偏上），故基线非 0.5
PUPIL_Y_BASE = 0.42
PUPIL_Y_TOL = 0.20
# 持续注视触发时间
GAZE_HOLD_SECONDS = 2.0
# ===================== 终端日志 =====================
# 终端状态报告打印间隔（秒）
LOG_INTERVAL = 1.0
# 是否启用 ANSI 彩色输出（重定向到文件请设 False）
USE_ANSI_COLOR = True
# ===================== 跟踪增强 =====================
# 远距离 / 快速运动跟丢时，启用 Face Detection 重新定位 ROI
ENABLE_ROI_FALLBACK = True
# Face Detection 置信度（越低越能侦测到远距离/小脸）
DETECT_CONFIDENCE = 0.3
# ROI 在 detection bbox 上四周扩展的比例
ROI_PAD_RATIO = 0.35
# ==================================================

# MediaPipe 人脸网格（带瞳孔关键点）—— 主跟踪器（全图模式）
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# MediaPipe 人脸网格 —— ROI 重检测器（容差低，专门用在 detection 给的小区域上）
face_mesh_roi = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# MediaPipe 人脸检测器（model_selection=1 覆盖 0~5m 远距离）
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=DETECT_CONFIDENCE,
)

# ---- Face Mesh 关键索引 ----
# 鼻尖
NOSE_TIP = 1
# 前额顶 / 下巴
FOREHEAD = 10
CHIN = 152
# 左眼外角 / 内角 / 上 / 下（注：mediapipe 视角，对应"人脸的左眼"）
L_OUTER, L_INNER = 33, 133
L_TOP, L_BOTTOM = 159, 145
# 右眼外角 / 内角 / 上 / 下
R_OUTER, R_INNER = 263, 362
R_TOP, R_BOTTOM = 386, 374
# 瞳孔
LEFT_PUPIL = 468
RIGHT_PUPIL = 473

# 左/右眼轮廓（画框用）
LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]


# ===================== 工具函数 =====================
def bbox_from_indices(landmarks, indices, w, h, pad=0):
    xs = [landmarks.landmark[i].x for i in indices]
    ys = [landmarks.landmark[i].y for i in indices]
    x1 = max(0, int(min(xs) * w) - pad)
    y1 = max(0, int(min(ys) * h) - pad)
    x2 = min(w - 1, int(max(xs) * w) + pad)
    y2 = min(h - 1, int(max(ys) * h) + pad)
    return x1, y1, x2, y2


def face_bbox(landmarks, w, h, pad=10):
    xs = [p.x for p in landmarks.landmark]
    ys = [p.y for p in landmarks.landmark]
    x1 = max(0, int(min(xs) * w) - pad)
    y1 = max(0, int(min(ys) * h) - pad)
    x2 = min(w - 1, int(max(xs) * w) + pad)
    y2 = min(h - 1, int(max(ys) * h) + pad)
    return x1, y1, x2, y2


def detect_face_roi(rgb, w, h, pad_ratio=ROI_PAD_RATIO):
    """用 Face Detection 在全图上找到最大人脸 ROI。

    返回 (x1, y1, x2, y2) 全图像素坐标，找不到返回 None。
    会向四周按 pad_ratio 扩展，给 face_mesh 一些余量。
    """
    det = face_detector.process(rgb)
    if not det.detections:
        return None
    best = max(det.detections, key=lambda d: d.score[0])
    rel = best.location_data.relative_bounding_box
    bx = rel.xmin * w
    by = rel.ymin * h
    bw = rel.width * w
    bh = rel.height * h
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    x1 = max(0, int(bx - pad_x))
    y1 = max(0, int(by - pad_y))
    x2 = min(w, int(bx + bw + pad_x))
    y2 = min(h, int(by + bh + pad_y))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return (x1, y1, x2, y2)


def remap_landmarks(landmarks, bbox, img_w, img_h):
    """把 ROI 内归一化坐标映射回全图归一化坐标（in-place 修改）。"""
    x1, y1, x2, y2 = bbox
    roi_w = x2 - x1
    roi_h = y2 - y1
    for lm in landmarks.landmark:
        lm.x = (lm.x * roi_w + x1) / img_w
        lm.y = (lm.y * roi_h + y1) / img_h


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ===================== 注视核心算法 =====================
def face_orientation(lm):
    """估算脸部朝向。

    返回:
        yaw   : 鼻尖相对双眼水平中线的偏移（归一化到眼宽，0 = 正脸）
        pitch : 鼻尖在 (前额 -> 下巴) 中的相对垂直位置（0.6~0.8 = 正脸）
    """
    l_out = lm.landmark[L_OUTER]
    r_out = lm.landmark[R_OUTER]
    nose = lm.landmark[NOSE_TIP]
    forehead = lm.landmark[FOREHEAD]
    chin = lm.landmark[CHIN]

    eyes_mid_x = (l_out.x + r_out.x) / 2
    eye_width = abs(r_out.x - l_out.x) + 1e-6
    yaw = (nose.x - eyes_mid_x) / eye_width  # 有符号，正=右偏

    face_height = abs(chin.y - forehead.y) + 1e-6
    pitch = (nose.y - forehead.y) / face_height  # 0=顶, 1=下巴

    return yaw, pitch


def pupil_centering(lm):
    """计算左右眼瞳孔水平/垂直归一化位置。

    返回:
        x_dev: 左右两眼水平偏差的最大值（更严格：一只眼偏就算不居中）
        y_dev: 左右两眼垂直偏差的最大值
    """
    # 左眼
    l_out = lm.landmark[L_OUTER]
    l_in = lm.landmark[L_INNER]
    l_top = lm.landmark[L_TOP]
    l_bot = lm.landmark[L_BOTTOM]
    l_pup = lm.landmark[LEFT_PUPIL]

    l_x_ratio = (l_pup.x - l_out.x) / ((l_in.x - l_out.x) + 1e-6)
    l_y_ratio = (l_pup.y - l_top.y) / ((l_bot.y - l_top.y) + 1e-6)

    # 右眼（注意 mediapipe 索引顺序：内角 362 在左侧，外角 263 在右侧）
    r_in = lm.landmark[R_INNER]
    r_out = lm.landmark[R_OUTER]
    r_top = lm.landmark[R_TOP]
    r_bot = lm.landmark[R_BOTTOM]
    r_pup = lm.landmark[RIGHT_PUPIL]

    r_x_ratio = (r_pup.x - r_in.x) / ((r_out.x - r_in.x) + 1e-6)
    r_y_ratio = (r_pup.y - r_top.y) / ((r_bot.y - r_top.y) + 1e-6)

    # 左右两眼分别取偏差，再取最大（避免一眼正一眼斜被均值"洗掉"）
    l_x_dev = abs(l_x_ratio - 0.5)
    r_x_dev = abs(r_x_ratio - 0.5)
    l_y_dev = abs(l_y_ratio - PUPIL_Y_BASE)
    r_y_dev = abs(r_y_ratio - PUPIL_Y_BASE)
    return max(l_x_dev, r_x_dev), max(l_y_dev, r_y_dev)


def evaluate_gaze(lm):
    """综合判定是否注视相机。

    返回字典:
        face_ok    : 脸朝相机
        pupil_ok   : 瞳孔居中
        looking    : 同时满足 (=瞬时注视)
        yaw / pitch / px_dev / py_dev : 调试用
    """
    yaw, pitch = face_orientation(lm)
    face_ok = (abs(yaw) < YAW_TOL) and (PITCH_MIN < pitch < PITCH_MAX)

    px_dev, py_dev = pupil_centering(lm)
    pupil_ok = (px_dev < PUPIL_X_TOL) and (py_dev < PUPIL_Y_TOL)

    return {
        "face_ok": face_ok,
        "pupil_ok": pupil_ok,
        "looking": face_ok and pupil_ok,
        "yaw": yaw,
        "pitch": pitch,
        "px_dev": px_dev,
        "py_dev": py_dev,
    }


class GazeHoldTimer:
    """瞬时注视 -> 持续 N 秒 -> 触发 locked 标志。"""

    def __init__(self, hold_seconds: float = GAZE_HOLD_SECONDS):
        self.hold = hold_seconds
        self.start_t = None
        self.locked = False

    def update(self, instant_looking: bool):
        now = time.time()
        if instant_looking:
            if self.start_t is None:
                self.start_t = now
            elapsed = now - self.start_t
            if elapsed >= self.hold:
                self.locked = True
            return elapsed, self.locked
        else:
            self.start_t = None
            self.locked = False
            return 0.0, False


# ===================== 系统监控（CPU / GPU） =====================
class SystemMonitor(threading.Thread):
    """后台线程定期采集 CPU 与 GPU 占用率。

    GPU 读取兼容三种方式（按优先级）：
      1) pynvml         -> PC 端 NVIDIA 显卡（最准）
      2) Jetson sysfs   -> /sys/devices/.../gpu*/load
      3) nvidia-smi     -> 命令行兜底
    任一不可用就自动 fallback 到下一种；都失败显示 N/A。
    """

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.cpu = 0.0
        self.gpu = None        # None 表示读取不到
        self.gpu_source = None  # 显示用："nvml" / "sysfs" / "smi"
        self._stop = False

        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)  # 预热
        except ImportError:
            pass

        # 准备 GPU 读取后端
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_source = "nvml"
        except Exception:
            self._nvml = None

        if self._nvml_handle is None:
            self._gpu_sysfs = self._find_jetson_gpu_path()
            if self._gpu_sysfs:
                self.gpu_source = "sysfs"

        if self._nvml_handle is None and not getattr(self, "_gpu_sysfs", None):
            if self._cmd_exists("nvidia-smi"):
                self.gpu_source = "smi"

    @staticmethod
    def _cmd_exists(cmd: str) -> bool:
        from shutil import which
        return which(cmd) is not None

    @staticmethod
    def _find_jetson_gpu_path():
        """查找 Jetson 的 GPU load 文件（仅浅层匹配，避免 sysfs 循环链接）。"""
        candidates = [
            "/sys/devices/gpu.0/load",
            "/sys/devices/platform/host1x/57000000.gpu/load",
            "/sys/devices/platform/host1x/15700000.gpu/load",
            "/sys/devices/57000000.gpu/load",
        ]
        # 浅层 glob（一层通配，不递归）
        for pattern in (
            "/sys/devices/gpu*/load",
            "/sys/devices/platform/*/load",
            "/sys/devices/platform/host1x/*gpu*/load",
        ):
            candidates += glob.glob(pattern)
        for p in candidates:
            try:
                if os.path.exists(p):
                    return p
            except OSError:
                continue
        return None

    def _read_gpu(self):
        # 1) pynvml
        if self._nvml_handle is not None:
            try:
                return float(
                    self._nvml.nvmlDeviceGetUtilizationRates(
                        self._nvml_handle
                    ).gpu
                )
            except Exception:
                return None
        # 2) Jetson sysfs
        path = getattr(self, "_gpu_sysfs", None)
        if path:
            try:
                with open(path, "r") as f:
                    raw = int(f.read().strip())
                # 不少 Jetson 上是 0~1000 千分制，>100 就按千分制处理
                return raw / 10.0 if raw > 100 else float(raw)
            except Exception:
                return None
        # 3) nvidia-smi
        if self.gpu_source == "smi":
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=1.0,
                ).decode().strip().splitlines()
                return float(out[0])
            except Exception:
                return None
        return None

    def run(self):
        while not self._stop:
            if self._psutil is not None:
                self.cpu = self._psutil.cpu_percent(interval=self.interval)
            else:
                time.sleep(self.interval)
            self.gpu = self._read_gpu()

    def stop(self):
        self._stop = True


def draw_top_right(frame, lines, scale=0.7, color=(0, 255, 0),
                   thickness=2, margin=12, line_gap=8):
    """在画面右上角逐行右对齐绘制。"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = margin + int(20 * scale * 1.5)
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = w - tw - margin
        # 半透明黑底（提升可读性）
        cv2.rectangle(
            frame,
            (x - 4, y - th - 4),
            (x + tw + 4, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)
        y += th + line_gap


# ===================== 终端日志（ANSI 彩色） =====================
_ANSI = {
    "reset":    "\033[0m",
    "bold":     "\033[1m",
    "dim":      "\033[2m",
    "red":      "\033[91m",
    "green":    "\033[92m",
    "yellow":   "\033[93m",
    "cyan":     "\033[96m",
    "magenta":  "\033[95m",
    "gray":     "\033[90m",
    "bg_green": "\033[42;30m",
    "bg_red":   "\033[41;97m",
}


def _c(text: str, *styles: str) -> str:
    """根据样式名组合 ANSI 颜色，重定向到文件时自动降级为纯文本。"""
    if not USE_ANSI_COLOR:
        return text
    return "".join(_ANSI[s] for s in styles) + text + _ANSI["reset"]


def _badge(ok: bool) -> str:
    """状态徽章：通过 -> 绿色 ' OK ' / 未通过 -> 红色 'MISS'。"""
    if ok:
        return _c(" OK ", "bold", "green")
    return _c("MISS", "bold", "red")


def print_status_report(info, used_roi, locked, elapsed, hold_target,
                        fps, cpu, gpu):
    """在终端打印一帧多行状态报告（每 LOG_INTERVAL 秒调用一次）。"""
    ts = time.strftime("%H:%M:%S")
    bar = _c("─" * 66, "gray")

    head_left = _c(f"[{ts}]", "cyan", "bold")
    gpu_s = f"{gpu:5.1f}%" if gpu is not None else " N/A "
    head_right = (
        f"FPS {fps:5.1f}   CPU {cpu:5.1f}%   GPU {gpu_s}"
    )
    roi_tag = f"   {_c('[ROI]', 'yellow', 'bold')}" if used_roi else ""

    print()
    print(bar)
    print(f" {head_left}   {head_right}{roi_tag}")
    print(bar)

    if info is None:
        print(
            f"  Face detected     {_badge(False)}   "
            + _c("no face in frame", "gray")
        )
        print(f"  Face orientation  {_badge(False)}")
        print(f"  Pupil centered    {_badge(False)}")
        print(f"  Gaze locked       {_badge(False)}")
    else:
        face_ok = info["face_ok"]
        pupil_ok = info["pupil_ok"]

        print(f"  Face detected     {_badge(True)}")
        print(
            f"  Face orientation  {_badge(face_ok)}   "
            f"yaw={info['yaw']:+.2f}  pitch={info['pitch']:.2f}"
        )
        print(
            f"  Pupil centered    {_badge(pupil_ok)}   "
            f"px={info['px_dev']:.2f}  py={info['py_dev']:.2f}"
        )
        if locked:
            big = _c("  LOOKING AT CAMERA  ", "bg_green", "bold")
            print(
                f"  Gaze locked       {_badge(True)}   "
                f">= {hold_target:.1f}s   {big}"
            )
        elif face_ok and pupil_ok:
            wait = _c("WAIT", "bold", "yellow")
            print(
                f"  Gaze locked       {wait}   "
                f"holding {elapsed:.1f}s / {hold_target:.1f}s"
            )
        else:
            print(f"  Gaze locked       {_badge(False)}")


# ===================== 主流程 =====================
def main():
    cap = open_camera()
    if not cap.isOpened():
        print("无法打开相机，请检查 CAM_ID 或线缆连接")
        return

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    print(f"相机已打开: {real_w}x{real_h} @ {real_fps:.1f} FPS  编码={fourcc_str}")
    print(f"持续 {GAZE_HOLD_SECONDS:.1f} 秒注视将触发 GAZE LOCKED")
    print("按 ESC 退出")

    timer = GazeHoldTimer(GAZE_HOLD_SECONDS)
    sysmon = SystemMonitor(interval=0.5)
    sysmon.start()
    print(f"系统监控: CPU=psutil  GPU={sysmon.gpu_source or 'N/A'}")
    print(_c(
        f"终端状态日志：每 {LOG_INTERVAL:.1f} 秒打印一次",
        "cyan", "bold",
    ))

    t0 = time.time()
    frames = 0
    fps_show = 0.0
    last_log_t = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        if frames >= 10:
            now = time.time()
            fps_show = frames / (now - t0)
            t0 = now
            frames = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)

        # ===== 跟丢兜底：用 Face Detection 找 ROI -> ROI 上跑 Face Mesh =====
        used_roi = False
        if ENABLE_ROI_FALLBACK and not res.multi_face_landmarks:
            bbox = detect_face_roi(rgb, real_w, real_h)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = rgb[y1:y2, x1:x2].copy()
                roi.flags.writeable = False
                res = face_mesh_roi.process(roi)
                if res.multi_face_landmarks:
                    for lms in res.multi_face_landmarks:
                        remap_landmarks(lms, bbox, real_w, real_h)
                    used_roi = True

        frame = cv2.resize(frame, SHOW_SIZE)
        frame.flags.writeable = True
        h, w = frame.shape[:2]

        info = None
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                info = evaluate_gaze(lm)
                # 颜色独立反映两个条件：
                #   - 人脸框 / "Face" 文字：脸朝相机 -> 绿，否则红
                #   - 瞳孔点：双眼瞳孔居中 -> 绿，否则红
                #   - 眼睛框：与瞳孔同步切换，便于一眼分辨
                face_color = (0, 255, 0) if info["face_ok"] else (0, 0, 255)
                pupil_color = (
                    (0, 255, 0) if info["pupil_ok"] else (0, 0, 255)
                )

                fx1, fy1, fx2, fy2 = face_bbox(lm, w, h, pad=12)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                cv2.putText(
                    frame, "Face", (fx1, max(0, fy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2,
                )

                lex1, ley1, lex2, ley2 = bbox_from_indices(
                    lm, LEFT_EYE_IDX, w, h, pad=6,
                )
                rex1, rey1, rex2, rey2 = bbox_from_indices(
                    lm, RIGHT_EYE_IDX, w, h, pad=6,
                )
                cv2.rectangle(
                    frame, (lex1, ley1), (lex2, ley2), pupil_color, 2,
                )
                cv2.rectangle(
                    frame, (rex1, rey1), (rex2, rey2), pupil_color, 2,
                )

                lx = int(lm.landmark[LEFT_PUPIL].x * w)
                ly = int(lm.landmark[LEFT_PUPIL].y * h)
                rx = int(lm.landmark[RIGHT_PUPIL].x * w)
                ry = int(lm.landmark[RIGHT_PUPIL].y * h)
                cv2.circle(frame, (lx, ly), 4, pupil_color, -1)
                cv2.circle(frame, (rx, ry), 4, pupil_color, -1)

        instant_looking = bool(info and info["looking"])
        elapsed, locked = timer.update(instant_looking)

        # ===== 顶部状态条 =====
        if info is None:
            status_text = "No face"
            status_color = (200, 200, 200)
        elif locked:
            status_text = "GAZE LOCKED  (>= 2.0s)"
            status_color = (0, 255, 0)
        elif instant_looking:
            status_text = (
                f"Holding... {elapsed:.1f}s / {GAZE_HOLD_SECONDS:.1f}s"
            )
            status_color = (0, 255, 255)
        else:
            reasons = []
            if not info["face_ok"]:
                reasons.append("face")
            if not info["pupil_ok"]:
                reasons.append("pupil")
            status_text = "Not looking (" + ",".join(reasons) + ")"
            status_color = (0, 0, 255)

        cv2.putText(
            frame, status_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3,
        )

        # ===== 右上角：FPS / CPU / GPU =====
        if sysmon.gpu is not None:
            gpu_text = f"GPU: {sysmon.gpu:5.1f}%"
        else:
            gpu_text = "GPU: N/A"
        draw_top_right(
            frame,
            [
                f"FPS: {fps_show:5.1f}",
                f"CPU: {sysmon.cpu:5.1f}%",
                gpu_text,
            ],
            scale=0.7,
            color=(0, 255, 0),
            thickness=2,
        )

        # ===== 调试信息（小字） =====
        if info is not None:
            roi_tag = " [ROI]" if used_roi else ""
            debug = (
                f"yaw={info['yaw']:+.2f}  pitch={info['pitch']:.2f}  "
                f"px={info['px_dev']:.2f}  py={info['py_dev']:.2f}  "
                f"FPS={fps_show:.1f}{roi_tag}"
            )
            cv2.putText(
                frame, debug, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # ===== 大字标识：锁定时居中显示 =====
        if locked:
            big = "LOOKING AT CAMERA"
            (tw, th), _ = cv2.getTextSize(
                big, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4,
            )
            cx = (w - tw) // 2
            cy = h - 60
            cv2.rectangle(
                frame, (cx - 15, cy - th - 15), (cx + tw + 15, cy + 15),
                (0, 255, 0), -1,
            )
            cv2.putText(
                frame, big, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4,
            )

        # ===== 终端定期状态报告 =====
        if time.time() - last_log_t >= LOG_INTERVAL:
            print_status_report(
                info, used_roi, locked, elapsed, GAZE_HOLD_SECONDS,
                fps_show, sysmon.cpu, sysmon.gpu,
            )
            last_log_t = time.time()

        cv2.imshow("Gaze Robot (Orin Nano)", frame)
        if cv2.waitKey(1) == 27:
            break

    sysmon.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
