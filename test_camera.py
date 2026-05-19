# ==============================================
# 相机测试脚本（2K 140°FOV 专用）
# 关键：UVC 相机必须设 MJPG 编码才能跑高帧率
# ==============================================

import time

import cv2

# ===================== 配置 =====================
WIDTH = 2560
HEIGHT = 1440
TARGET_FPS = 30
CAM_ID = 0  # USB相机一般是 0 或 1，CSI相机是 0
USE_MJPG = True  # 高分辨率必须 True，否则会掉到 1~5 FPS
# =================================================


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

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(
        [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
    )
    print(f"相机已打开: {real_w}x{real_h} @ {real_fps:.1f} FPS  编码={fourcc_str}")
    print("按 ESC 退出")

    # 实时 FPS 统计
    t0 = time.time()
    frames = 0
    fps_show = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("获取画面失败")
            break

        frames += 1
        if frames >= 10:
            now = time.time()
            fps_show = frames / (now - t0)
            t0 = now
            frames = 0

        show = cv2.resize(frame, (1280, 720))
        cv2.putText(
            show,
            f"FPS: {fps_show:.1f}  {real_w}x{real_h} {fourcc_str}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera 2K", show)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
