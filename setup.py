# ==============================================
# Jetson Orin Nano 注视识别工程 - 安装脚本
# 用法:
#   pip install -e .        # 开发模式安装（推荐）
#   pip install .           # 普通安装
# 安装完成后可直接使用以下命令:
#   gaze-camera-test        # 测试相机
#   gaze-robot              # 运行注视识别
# ==============================================

from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent


def read_requirements() -> list:
    """从 requirements.txt 读取依赖列表。"""
    req_file = HERE / "requirements.txt"
    if not req_file.exists():
        return []
    return [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="gaze_robot",
    version="0.1.0",
    description="Jetson Orin Nano 注视识别工程（人脸 + 瞳孔 + 是否注视相机）",
    author="bird",
    license="MIT",
    python_requires=">=3.8",
    py_modules=["gaze_robot", "test_camera"],
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "gaze-robot=gaze_robot:main",
            "gaze-camera-test=test_camera:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
