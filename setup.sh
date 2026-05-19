#!/bin/bash
# ==============================================
# Jetson Orin Nano 一键环境安装脚本
# 环境名：atten_ident   Python：3.8
# ==============================================
set -e

ENV_NAME="atten_ident"

echo "[1/3] 创建 conda 环境：${ENV_NAME} (python=3.8)"
conda create -n ${ENV_NAME} python=3.8 -y

echo "[2/3] 激活环境并安装依赖"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

pip install --upgrade pip
pip install numpy "opencv-python==4.8.1.78" pillow
pip install "mediapipe==0.10.9"
pip install onnxruntime

echo "[3/3] 安装完成。使用方式："
echo "  conda activate ${ENV_NAME}"
echo "  python test_camera.py     # 测试相机"
echo "  python gaze_robot.py      # 运行注视识别"
