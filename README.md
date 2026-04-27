# MuseTalk — Live Avatar Pipeline

> **Fork of [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)**  
> Extended with a real-time, browser-based talking-head avatar pipeline: **LLM → Kokoro TTS → MuseTalk → browser canvas**.

---

## What This Fork Adds

The upstream MuseTalk repo provides offline video-dubbing inference. This fork adds a **fully streaming, conversational avatar system** on top of it:

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
┌─────────────────┐  text_q   ┌─────────────────┐  tts_q   ┌─────────────────────────────┐
│  LLM thread     │ ────────► │   TTS thread     │ ───────► │  Whisper + UNet worker      │
│  (per request)  │           │ (always 1 ahead) │          │  (GPU inference)            │
└─────────────────┘           └─────────────────┘          └──────────────┬──────────────┘
                                                                           │
                                                              ┌────────────▼────────────┐
                                                              │  Blend pool             │
                                                              │  (ThreadPoolExecutor)   │
                                                              │  CPU blends batch N     │
                                                              │  while GPU runs N+1     │
                                                              └────────────┬────────────┘
                                                                           │ SyncedChunk
                                                                           │ (audio + frames)
                                                                           ▼
                                                              ┌─────────────────────────┐
                                                              │  Flask SSE /sync_feed   │
                                                              │  parallel JPEG encode   │
                                                              └────────────┬────────────┘
                                                                           │
                                                                           ▼
                                                              Browser (canvas + Web Audio)
                                                              AudioContext as master clock
                                                              lip-sync via playbackRate
```

### Thread Model

| Thread | Queue in | Queue out | Work |
|---|---|---|---|
| **LLM stream** | — | `text_q` | Streams tokens, flushes complete sentences |
| **TTS** | `text_q` | `tts_q` | Kokoro synthesis for sentence N+1 while UNet runs N |
| **Whisper + UNet** | `tts_q` | `output_q` | Whisper feature extraction → UNet inference → parallel blend |
| **Flask SSE** | `output_q` | browser | Parallel JPEG encode → base64 → Server-Sent Event |

The TTS thread stays **one sentence ahead** of the UNet worker. By the time UNet finishes sentence N, audio for sentence N+1 is already ready — Whisper and UNet never wait on synthesis.

### A/V Sync Strategy

Audio is the master clock. The browser schedules each audio chunk with `AudioContext.currentTime`, then plays frames at a rate derived from `audio_duration / frame_count`. This eliminates JavaScript timer drift entirely.

---

## Latency Optimisations

| Optimisation | Where | Saving |
|---|---|---|
| Dedicated TTS thread (runs ahead) | `musetalk_avatar_pipeline.py` | ~100–200 ms overlap per sentence |
| Parallel frame blending — CPU blends batch N while GPU runs N+1 | `_blend_batch()` + `ThreadPoolExecutor` | ~200–300 ms |
| In-memory audio — `BytesIO` passed directly to `AudioProcessor`, no temp file | `audio_processor.get_audio_feature()` | ~10–20 ms |
| Polyphase resample (`resample_poly`) instead of DFT-based `resample` | `tts_kokoro._resample()` | ~25 ms |
| Parallel JPEG encoding via `ThreadPoolExecutor` | `run_musetalk_avatar._encode_frame()` | ~80–100 ms |
| `torch.compile(backend='cudagraphs')` on UNet | pipeline init | ~20–40% per batch after warmup |
| `torch.backends.cudnn.benchmark = True` | pipeline init | minor, fixed-size conv inputs |

---

## Installation

### Prerequisites

- Python 3.10
- CUDA 11.8
- [uv](https://docs.astral.sh/uv/) package manager
- FFmpeg + espeak-ng

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install system deps
sudo apt-get install -y ffmpeg espeak-ng
```

### Steps

**1. Clone the repo**

```bash
git clone https://github.com/sarangKP/Avatar.git
cd Avatar
```

**2. Install Python dependencies**

```bash
uv sync
source .venv/bin/activate
```

**3. Install MMLab packages**

MMLab wheels require CUDA-specific pre-built binaries and cannot go through `uv`. Run after `uv sync`:

```bash
bash setup_mmlab.sh
```

Installs: `mmengine`, `mmcv==2.0.1`, `mmdet==3.1.0`, `mmpose==1.1.0`

**4. Download model weights**

```bash
bash download_weights.sh
```

Expected layout:

```
models/
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
```

> **Note:** `79999_iter.pth` may fail to download automatically due to Google Drive restrictions. If so, download it manually via browser and copy it to `models/face-parse-bisent/79999_iter.pth`.

---

## Running

### Option A — SSH port forward (recommended)

On your local machine:

```bash
ssh -L 7860:localhost:7860 user@<server-ip>
```

On the server:

```bash
python run_musetalk_avatar.py \
    --avatar_image examples/face_1.png \
    --llm_backend echo \
    --port 7860
```

Open `http://localhost:7860` in your local browser.

### Option B — Direct access

The server binds to `0.0.0.0` by default:

```bash
python run_musetalk_avatar.py \
    --avatar_image examples/face_1.png \
    --llm_backend echo \
    --port 7860
```

Open `http://<server-ip>:7860`.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--avatar_image` | *(required)* | Path to reference face image (PNG/JPG) |
| `--unet_config` | `./models/musetalkV15/musetalk.json` | UNet config path |
| `--unet_model_path` | `./models/musetalkV15/unet.pth` | UNet weights path |
| `--whisper_dir` | `./models/whisper` | Whisper feature extractor directory |
| `--vae_type` | `sd-vae` | VAE type |
| `--use_float16` | `True` | fp16 inference (recommended for RTX 3080) |
| `--batch_size` | `8` | Frames per UNet batch |
| `--bbox_shift` | `0` | Vertical shift for mouth crop (px) |
| `--extra_margin` | `10` | Extra pixels around face crop |
| `--fps` | `25` | Output frame rate |
| `--tts_voice` | `af_heart` | Kokoro voice tag |
| `--tts_speed` | `1.0` | TTS speech rate multiplier |
| `--llm_backend` | `echo` | `echo` / `openai` / `ollama` |
| `--llm_model` | `llama3.2` | LLM model name |
| `--llm_api_key` | `None` | API key (OpenAI / compatible) |
| `--llm_base_url` | `None` | Custom API base URL |
| `--ollama_host` | `http://localhost:11434` | Ollama server URL |
| `--port` | `7860` | Server port |

### LLM Backend Examples

```bash
# Local Ollama (zero cost)
python run_musetalk_avatar.py --avatar_image face.jpg \
    --llm_backend ollama --llm_model llama3.2

# OpenAI
python run_musetalk_avatar.py --avatar_image face.jpg \
    --llm_backend openai --llm_model gpt-4o-mini --llm_api_key sk-...

# Echo — reflects input directly, no LLM required (good for pipeline testing)
python run_musetalk_avatar.py --avatar_image face.jpg --llm_backend echo
```

---

## Avatar Pre-processing

On startup the pipeline pre-processes the reference image once:

1. Detects face landmarks and bounding box via MMPose + FaceAlignment
2. Crops and resizes the face region to 256×256
3. Encodes the crop through the VAE to produce a reference latent
4. Computes blending masks via the face-parsing model

This is done once — **no per-frame avatar encoding** during inference. Only the audio-conditioned UNet runs per batch.

---

## Optional: Face Enhancement (GFPGAN)

`musetalk/utils/enhancer.py` applies GFPGAN face restoration per frame after VAE decode and blending. When `gfpgan` is not installed, `enhance_frame()` returns the original frame unchanged — no errors.

To enable:

```bash
pip install gfpgan
# weights at: experiments/pretrained_models/GFPGANv1.4.pth
```

---

## Project Structure

```
.
├── run_musetalk_avatar.py        # Live avatar server (entry point)
├── musetalk_avatar_pipeline.py   # Core streaming pipeline
├── tts_kokoro.py                 # Kokoro TTS wrapper
├── llm_wrapper.py                # Streaming LLM (Ollama / OpenAI / Echo)
├── musetalk/
│   └── utils/
│       ├── audio_processor.py    # Whisper feature extraction (accepts BytesIO)
│       ├── blending.py           # Frame compositing
│       ├── enhancer.py           # GFPGAN post-processing (optional)
│       ├── face_parsing.py
│       ├── preprocessing.py
│       └── utils.py
├── models/                       # Downloaded weights
├── experiments/pretrained_models/
│   └── GFPGANv1.4.pth            # Optional GFPGAN weights
├── scripts/inference.py
├── configs/
├── download_weights.sh
├── setup_mmlab.sh
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

fp16 mode (`--use_float16`) is strongly recommended on consumer GPUs. On an RTX 3080 with batch size 8, the pipeline sustains real-time 25 fps output after the first-inference CUDA graph warmup (~25s one-time cost).

---

## Original MuseTalk Inference (Offline)

Standard offline inference from the upstream repo still works:

```bash
# MuseTalk 1.5 (recommended)
sh inference.sh v1.5 normal

# MuseTalk 1.0
sh inference.sh v1.0 normal
```

### Gradio Demo

```bash
python app.py --use_float16 --ffmpeg_path /path/to/ffmpeg
```

### Training

```bash
python -m scripts.preprocess --config ./configs/training/preprocess.yaml
sh train.sh stage1
sh train.sh stage2
```

---

## Acknowledgements

- [MuseTalk (TMElyralab)](https://github.com/TMElyralab/MuseTalk) — base model and architecture
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — text-to-speech synthesis
- [GFPGAN (TencentARC)](https://github.com/TencentARC/GFPGAN) — face restoration
- [Whisper (OpenAI)](https://github.com/openai/whisper) — audio feature extraction
- [DWPose](https://github.com/IDEA-Research/DWPose), [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch), [face-alignment](https://github.com/1adrianb/face-alignment)

---

## Citation

```bibtex
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and
          Zeng, Yubin and Chao, Zhan and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```
