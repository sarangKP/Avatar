# MuseTalk — Live Avatar Pipeline

> **Fork of [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)**  
> Extended with a real-time, browser-based talking-head avatar pipeline: **LLM → Kokoro TTS → MuseTalk → browser canvas**.

---

## What's Added in This Fork

The original MuseTalk repo provides offline video-dubbing inference. This fork adds a **fully streaming, conversational avatar system** built on top of it:

| File | Role |
|---|---|
| `run_musetalk_avatar.py` | Flask server — entry point for the live avatar |
| `musetalk_avatar_pipeline.py` | Core pipeline: LLM → TTS → MuseTalk → output queue |
| `tts_kokoro.py` | Kokoro TTS wrapper (24 kHz → 16 kHz resampling) |
| `llm_wrapper.py` | Streaming LLM backend (Ollama / OpenAI / Echo) |
| `musetalk/utils/enhancer.py` | Optional GFPGAN face-restoration post-processing |

---

## Architecture

```
User text input (browser)
        │
        ▼
┌─────────────────┐   sentence stream  ┌──────────────────────────────┐
│  LLM thread     │ ─────────────────► │  TTS + MuseTalk worker       │
│  (streaming)    │                    │  (serial, per sentence)      │
└─────────────────┘                    └──────────────┬───────────────┘
                                                      │ SyncedChunk
                                                      │ (audio + frames)
                                                      ▼
                                            ┌───────────────────┐
                                            │Flask SSE/sync_feed│
                                            └────────┬──────────┘
                                                     │
                                                     ▼
                                            Browser (canvas + Web Audio)
                                            AudioContext as master clock
                                            lip-sync via playbackRate
```

### Thread Model

- **LLM thread** — streams tokens from the LLM backend, flushes complete sentences into `text_q`
- **TTS + MuseTalk worker** — picks sentences from `text_q`, synthesizes audio with Kokoro, runs MuseTalk UNet inference, blends frames, optionally enhances with GFPGAN, pushes `SyncedChunk` objects to `output_q`
- **Flask SSE** — reads `output_q` and streams each chunk to the browser as a Server-Sent Event containing base64-encoded WAV audio + JPEG frames
- **Browser** — decodes chunks, schedules audio via `AudioContext`, renders frames on `<canvas>`, adjusts `playbackRate` to maintain A/V sync

### A/V Sync Strategy

Audio playback time is the master clock. The browser schedules each audio chunk with `AudioContext.currentTime`, then plays back the corresponding frames at a rate derived from `audio_duration / frame_count`. This eliminates JavaScript timer drift entirely.

---

## Installation

### Prerequisites

- Python 3.10
- CUDA 11.8
- [uv](https://docs.astral.sh/uv/) package manager
- FFmpeg

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install FFmpeg
sudo apt-get install -y ffmpeg espeak-ng

---
1. Clone the repo

git clone https://github.com/sarangKP/Avatar.git
cd Avatar

2. Install Python dependencies

# Creates .venv automatically and installs all pip-installable deps
uv sync

# Activate the venv
source .venv/bin/activate

3. Install MMLab packages

MMLab packages require CUDA-specific pre-built wheels and cannot go through uv.
Run the provided script after uv sync:

bash setup_mmlab.sh

This installs: mmengine, mmcv==2.0.1, mmdet==3.1.0, mmpose==1.1.0
4. Download model weights

bash download_weights.sh

Expected layout after download:

./models/
├── musetalkV15/
│   ├── musetalk.json
│   └── unet.pth
├── musetalk/
│   ├── musetalk.json
│   └── pytorch_model.bin
├── whisper/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── preprocessor_config.json
├── dwpose/
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent/
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── syncnet/
    └── latentsync_syncnet.pt

---
### Test run plan for SSH
Since you're on SSH (no browser on the server), you have two options:
**Option A — SSH port forward (recommended)**
On your local machine, open a new terminal and run:
```bash
ssh -L 7860:localhost:7860 innovation@<your-server-ip>

Then on the server, start with the echo backend (no LLM needed, reflects your input directly — good for pipeline testing):

python run_musetalk_avatar.py \
    --avatar_image examples/face_1.png \
    --llm_backend echo \
    --port 7860

Then open http://localhost:7860 in your local browser.

Option B — bind to all interfaces (if your server IP is directly reachable):

The server already binds to `0.0.0.0` by default, so just start it normally:

python run_musetalk_avatar.py \
    --avatar_image examples/face_1.png \
    --llm_backend echo \
    --port 7860

Then open http://<server-ip>:7860 in your browser.


### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--avatar_image` | *(required)* | Path to reference face image (PNG/JPG) |
| `--unet_config` | `./models/musetalkV15/musetalk.json` | MuseTalk UNet config |
| `--unet_model_path` | `./models/musetalkV15/unet.pth` | MuseTalk UNet weights |
| `--whisper_dir` | `./models/whisper` | Whisper feature extractor path |
| `--vae_type` | `sd-vae` | VAE type |
| `--use_float16` | `True` | Use fp16 (recommended for 3080) |
| `--batch_size` | `8` | Frames per UNet batch |
| `--bbox_shift` | `0` | Vertical shift for mouth crop (px) |
| `--extra_margin` | `10` | Extra pixels around face crop |
| `--fps` | `25` | Output frame rate |
| `--tts_voice` | `af_heart` | Kokoro voice tag |
| `--tts_speed` | `1.0` | TTS speech rate |
| `--llm_backend` | `echo` | `echo` / `openai` / `ollama` |
| `--llm_model` | `llama3.2` | LLM model name |
| `--llm_api_key` | `None` | API key (OpenAI / compatible) |
| `--llm_base_url` | `None` | Custom API base URL |
| `--ollama_host` | `http://localhost:11434` | Ollama server URL |
| `--port` | `7860` | Flask server port |

### LLM Backend Examples

```bash
# Local Ollama (zero cost)
python run_musetalk_avatar.py --avatar_image face.jpg --llm_backend ollama --llm_model llama3.2

# OpenAI
python run_musetalk_avatar.py --avatar_image face.jpg --llm_backend openai \
    --llm_model gpt-4o-mini --llm_api_key sk-...

# Echo (test pipeline without an LLM — reflects user input directly)
python run_musetalk_avatar.py --avatar_image face.jpg --llm_backend echo
```

---

## Avatar Pre-processing

On startup the pipeline pre-processes the reference image once:
1. Detects face landmarks and bounding box
2. Crops and resizes the face region to 256×256
3. Encodes the crop through the VAE to get a reference latent
4. Prepares blending masks via the face-parsing model

This pre-computation means **no per-frame avatar encoding** at inference time — only the audio-conditioned UNet runs per batch.

---

## Optional: Face Enhancement (GFPGAN)

The `musetalk/utils/enhancer.py` module applies GFPGAN face restoration as a post-processing step after the full VAE decode + blending pipeline. Enhancement runs in pixel space and is optional.

To enable it, ensure the GFPGAN weights are downloaded (see Installation §7) and the enhancer is imported in your inference script:

```python
from musetalk.utils.enhancer import enhance_frame
# called per-frame after get_image_blending()
frame = enhance_frame(frame)
```

---

## Original MuseTalk Inference (Offline)

Standard offline inference from the original repo still works:

```bash
# MuseTalk 1.5 (recommended)
sh inference.sh v1.5 normal

# MuseTalk 1.0
sh inference.sh v1.0 normal
```

See the original [Getting Started](#original-getting-started) section below for Gradio demo and training instructions.

---

## Project Structure

```
.
├── run_musetalk_avatar.py        # Live avatar Flask server (entry point)
├── musetalk_avatar_pipeline.py   # Core streaming pipeline
├── tts_kokoro.py                 # Kokoro TTS wrapper
├── llm_wrapper.py                # Streaming LLM (Ollama / OpenAI / Echo)
├── musetalk/
│   └── utils/
│       ├── enhancer.py           # GFPGAN post-processing (added)
│       ├── blending.py
│       ├── audio_processor.py
│       ├── face_parsing.py
│       ├── preprocessing.py
│       └── utils.py
├── models/                       # Downloaded weights (see above)
├── experiments/
│   └── pretrained_models/
│       └── GFPGANv1.4.pth        # Optional GFPGAN weights
├── scripts/
│   └── inference.py
├── configs/
├── requirements.txt
├── download_weights.sh
└── app.py                        # Original Gradio demo
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3050 Ti (4 GB VRAM) | RTX 3080 (10 GB VRAM) |
| RAM | 16 GB | 32 GB |
| CUDA | 11.7 | 11.8 |
| Python | 3.10 | 3.10 |

fp16 mode (`--use_float16`) is strongly recommended on consumer GPUs. On an RTX 3080, the pipeline sustains real-time 25 fps output with batch size 8.

---

## Original Getting Started

> The sections below are preserved from the upstream MuseTalk repo for reference.

**[github](https://github.com/TMElyralab/MuseTalk)** | **[huggingface](https://huggingface.co/TMElyralab/MuseTalk)** | **[Technical report](https://arxiv.org/abs/2410.10122)**

MuseTalk is a real-time high-quality audio-driven lip-syncing model (30 fps+ on Tesla V100), operating in the latent space of `ft-mse-vae`. It modifies a face region of 256×256 px, supports multiple languages, and is distinct from diffusion models — it inpaints in a single UNet step.

### Gradio Demo

```bash
python app.py --use_float16 --ffmpeg_path /path/to/ffmpeg
```

### Training

```bash
# Data preprocessing
python -m scripts.preprocess --config ./configs/training/preprocess.yaml

# Stage 1
sh train.sh stage1

# Stage 2
sh train.sh stage2
```

---

## Acknowledgements

- [MuseTalk (TMElyralab)](https://github.com/TMElyralab/MuseTalk) — base model and architecture
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — text-to-speech synthesis
- [GFPGAN (TencentARC)](https://github.com/TencentARC/GFPGAN) — face restoration
- [whisper (OpenAI)](https://github.com/openai/whisper) — audio feature extraction
- [dwpose](https://github.com/IDEA-Research/DWPose), [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch), [face-alignment](https://github.com/1adrianb/face-alignment)

---

## Citation

```bibtex
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and
          Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```