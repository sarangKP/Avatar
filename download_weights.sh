#!/bin/bash

CheckpointsDir="models"

mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

export HF_ENDPOINT=https://hf-mirror.com

echo ">>> Downloading MuseTalk V1.0 weights..."
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

echo ">>> Downloading MuseTalk V1.5 weights..."
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

echo ">>> Downloading SD VAE weights..."
hf download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

echo ">>> Downloading Whisper weights..."
hf download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

echo ">>> Downloading DWPose weights..."
hf download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

echo ">>> Downloading SyncNet weights..."
hf download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

echo ">>> Downloading Face Parse Bisent weights..."
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ All weights downloaded successfully!"