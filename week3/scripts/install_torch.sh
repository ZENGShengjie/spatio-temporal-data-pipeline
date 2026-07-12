#!/bin/bash
# 重装 torch 兼容 driver 595
set -e
echo "[1/3] uninstall torch"
pip uninstall -y torch torchvision torchaudio torch-geometric 2>&1 | tail -3
echo "[2/3] install torch 2.4 cu124"
pip install --user --break-system-packages \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
echo "[3/3] install torch-geometric"
pip install --user --break-system-packages torch-geometric==2.6.1 2>&1 | tail -5
