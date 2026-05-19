# ==============================================
# 相机测试脚本
# 支持两种相机：
#   1) 普通 2K UVC 广角相机 → USE_MJPG=True
#   2) ZED Mini 双目相机   → ZED_STEREO=True + USE_MJPG=False (只支持 YUYV)
# ==============================================

import os
import subprocess
import time

import cv2
import numpy as np

# ===================== 配置 =====================
# ZED Mini 可选分辨率（左右拼接，单眼是其一半宽度）:
#   4416x1242 @ 15 FPS         → 单眼 2208x1242（2.2K，最高分辨率）
#   3840x1080 @ 30/15 FPS      → 单眼 1920x1080（1080p）
#   2560x720  @ 60/30/15 FPS   → 单眼 1280x720（720p）
#   1344x376  @ 100/60/30 FPS  → 单眼 672x376（WVGA，低延迟）
WIDTH = 4416
HEIGHT = 1242
TARGET_FPS = 15
CAM_ID = 0
USE_MJPG = False  # ZED Mini 不支持 MJPG，必须 False（走 YUYV）
ZED_STEREO = True  # ZED Mini 双目拼接，需要裁掉右半边，只用左眼
FULLSCREEN = True  # 是否全屏显示（等比缩放，画面外用黑边填充）
WINDOW_NAME = "ZED Mini Test (left eye)"
# =================================================


def detect_screen_size(default=(1920, 1080)):
    """通过 xrandr 自动检测主屏分辨率，失败时返回默认值。"""
    try:
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")
        out = subprocess.check_output(
            ["xrandr"], env=env, stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        for line in out.splitlines():
            if "*" in line:
                token = line.strip().split()[0]  # 形如 1920x1080
                w, h = token.split("x")
                return int(w), int(h)
    except Exception:
        pass
    return default


def fit_letterbox(img, target_w, target_h):
    """等比缩放到 target，多余区域填黑色（letterbox）。"""
    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def open_camera() -> cv2.VideoCapture:
    """打开相机并做 Jetson/UVC 友好的参数配置。

    设置顺序很重要：FOURCC -> 分辨率 -> FPS -> Buffer。
    """
    # CAP_V4L2 在 Linux/Jetson 上更稳，避免走 GStreamer 默认管线
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)

    if USE_MJPG:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    cap = open_camera()
    if not cap.isOpened():
        print("无法打开相机，请检查 CAM_ID 或线缆连接")
        return

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(
        [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
    )
    real_w = raw_w // 2 if ZED_STEREO else raw_w
    real_h = raw_h
    src_tag = "ZED-LEFT" if ZED_STEREO else "MONO"
    print(
        f"相机原始流: {raw_w}x{raw_h} @ {real_fps:.1f} FPS  编码={fourcc_str}"
    )
    print(f"实际使用: {real_w}x{real_h} ({src_tag})")

    screen_w, screen_h = detect_screen_size()
    print(f"显示器分辨率: {screen_w}x{screen_h}  全屏={FULLSCREEN}")
    print("按 ESC 或 q 退出，按 F 切换全屏")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    is_fullscreen = FULLSCREEN

    t0 = time.time()
    frames = 0
    fps_show = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("获取画面失败")
            break

        if ZED_STEREO:
            frame = frame[:, : frame.shape[1] // 2]

        frames += 1
        if frames >= 10:
            now = time.time()
            fps_show = frames / (now - t0)
            t0 = now
            frames = 0

        show = fit_letterbox(frame, screen_w, screen_h)
        cv2.putText(
            show,
            f"FPS: {fps_show:.1f}  {real_w}x{real_h} {fourcc_str} {src_tag}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW_NAME, show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("f"):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
