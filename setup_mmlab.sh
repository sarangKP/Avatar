#!/bin/bash
# MMLab packages require CUDA-specific pre-built wheels.
# Uses direct pip install instead of mim to avoid pkg_resources issues.
# Run this AFTER: uv sync

set -e

echo ">>> Installing mmengine..."
pip install mmengine
echo "✅ mmengine installed."

echo ">>> Installing mmcv==2.0.1 (CUDA 11.8 + torch 2.0 wheel)..."
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
echo "✅ mmcv installed."

echo ">>> Installing mmdet==3.1.0..."
pip install mmdet==3.1.0
echo "✅ mmdet installed."

echo ">>> Installing chumpy (required by mmpose, needs --no-build-isolation)..."
pip install chumpy --no-build-isolation
echo "✅ chumpy installed."

echo ">>> Installing mmpose==1.1.0..."
pip install mmpose==1.1.0
echo "✅ mmpose installed."

echo ""
echo ">>> Installing system dependency for Kokoro TTS..."
sudo apt-get install -y espeak-ng
echo "✅ espeak-ng installed."

echo ""
echo "✅ All MMLab packages installed successfully."