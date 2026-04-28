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
| **LLM stream** | — | `text_q` | Streams tokens, flushes sentences, sub-divides into phrase chunks at clause boundaries |
| **TTS** | `text_q` | `tts_q` | Kokoro synthesis for chunk N+1 while UNet runs chunk N |
| **Whisper + UNet** | `tts_q` | `output_q` | Whisper feature extraction → UNet inference → parallel blend |
| **Flask SSE** | `output_q` | browser | Parallel JPEG encode → base64 → Server-Sent Event |

The TTS thread stays **one chunk ahead** of the UNet worker. By the time UNet finishes chunk N, audio for chunk N+1 is already synthesised — Whisper and UNet never wait on TTS.

### A/V Sync Strategy

Audio is the master clock. The browser schedules each audio chunk with `AudioContext.currentTime`, then plays frames at a rate derived from `audio_duration / frame_count`. This eliminates JavaScript timer drift entirely.

---

## Latency Optimisations

| Optimisation | Where | Saving |
|---|---|---|
| **Sub-sentence chunking** — clauses become independent `SyncedChunk`s | `_chunk_sentence()` in pipeline | **~600-800ms time-to-first-frame** |
| Dedicated TTS thread (runs ahead) | `musetalk_avatar_pipeline.py` | ~100–200 ms overlap per chunk |
| Parallel frame blending — CPU blends batch N while GPU runs N+1 | `_blend_batch()` + `ThreadPoolExecutor` | ~200–300 ms |
| Pre-processed avatar disk cache (hash-keyed) | `_avatar_cache_key()` | ~2-5 s on every restart after the first |
| In-memory audio — `BytesIO` passed directly to `AudioProcessor`, no temp file | `audio_processor.get_audio_feature()` | ~10–20 ms |
| Polyphase resample (`resample_poly`) instead of DFT-based `resample` | `tts_kokoro._resample()` | ~25 ms |
| Parallel JPEG encoding via `ThreadPoolExecutor` | `run_musetalk_avatar._encode_frame()` | ~80–100 ms |
| `torch.compile(backend='cudagraphs')` on UNet | pipeline init | ~20–40% per batch after warmup |
| `torch.backends.cudnn.benchmark = True` | pipeline init | minor, fixed-size conv inputs |

### Why sub-sentence chunking matters

Before: every sentence is one indivisible unit through TTS → Whisper → UNet → blend → emit. The user sees a frozen face for ~1.0–1.5 s before *any* lips move, and there's a hard cut between sentences in long replies.

After: sentences are split at clause boundaries (`,` `;` `:` `—`) when the chunk meets `min_chars`. Each phrase emits as soon as it's ready, so:
- **Time-to-first-frame** drops from ~1.2 s → **~400 ms**
- **Long paragraphs flow naturally** with breathing pauses at commas
- **GPU stays busy** — by the time chunk N+1 needs UNet, the worker is free

---

## Configuration

Runtime config lives in `config.yaml` at repo root. Edit + restart, or POST to `/config` for live updates.

```yaml
tts:
  voice: am_michael        # male: am_adam, am_michael, bm_george, bm_lewis
                           # female: af_heart, af_sky, bf_emma, bf_isabella
  speed: 1.0               # 0.5–2.0
  language: a              # 'a' = American English, 'b' = British English

enhancer:
  enabled: true            # GFPGAN face restoration (requires `uv add gfpgan`)

chunking:
  enabled: true            # phrase-level streaming for low time-to-first-frame
  min_chars: 20            # min chars before a clause boundary flushes a chunk

avatar:
  cache_dir: ./cache/avatars

unet:
  batch_size: 8
  fps: 25
  use_float16: true
  compile: true            # torch.compile (cudagraphs) — ~25s warmup, then 20-40% faster
```

### Live config endpoint

```bash
# Read current config
curl http://localhost:7860/config

# Switch to a different voice without restart
curl -X POST http://localhost:7860/config \
     -H 'Content-Type: application/json' \
     -d '{"tts": {"voice": "bm_george"}}'

# Toggle GFPGAN enhancement
curl -X POST http://localhost:7860/config \
     -H 'Content-Type: application/json' \
     -d '{"enhancer": {"enabled": false}}'

# Tune chunking
curl -X POST http://localhost:7860/config \
     -H 'Content-Type: application/json' \
     -d '{"chunking": {"enabled": true, "min_chars": 30}}'
```

Live-updatable: `tts.voice`, `tts.speed`, `tts.language`, `chunking.*`, `enhancer.enabled`.  
Restart-only: `unet.*`, `avatar.*`, `server.*`.

### Optional: GFPGAN for sharper mouth

GFPGAN restores fine facial details that the VAE blurs, especially around the mouth. To enable:

```bash
uv add gfpgan

# Download v1.4 weights (recommended)
wget -P experiments/pretrained_models/ \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

v1.3 weights also work — update `enhancer.model_path` in `config.yaml` to point to your file. It adds ~30–50 ms per frame. If `gfpgan` isn't installed, the pipeline silently skips enhancement (warns once in the log).

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

Most settings live in `config.yaml`. CLI flags override the config when explicitly provided.

| Argument | Default | Description |
|---|---|---|
| `--avatar_image` | *(required)* | Path to reference face image (PNG/JPG) |
| `--unet_config` | `./models/musetalkV15/musetalk.json` | UNet config path |
| `--unet_model_path` | `./models/musetalkV15/unet.pth` | UNet weights path |
| `--whisper_dir` | `./models/whisper` | Whisper feature extractor directory |
| `--vae_type` | `sd-vae` | VAE type |
| `--tts_voice` | config.yaml | Override `tts.voice` from config |
| `--tts_speed` | config.yaml | Override `tts.speed` from config |
| `--batch_size` | config.yaml | Override `unet.batch_size` from config |
| `--bbox_shift` | config.yaml | Override `avatar.bbox_shift` from config |
| `--extra_margin` | config.yaml | Override `avatar.extra_margin` from config |
| `--fps` | config.yaml | Override `unet.fps` from config |
| `--port` | config.yaml | Override `server.port` from config |
| `--llm_backend` | `echo` | `echo` / `openai` / `ollama` |
| `--llm_model` | `llama3.2` | LLM model name |
| `--llm_api_key` | `None` | API key (OpenAI / compatible) |
| `--llm_base_url` | `None` | Custom API base URL |
| `--ollama_host` | `http://localhost:11434` | Ollama server URL |

Settings only in `config.yaml` (no CLI flag): `use_float16`, `compile`, `tts.language`, `chunking.*`, `enhancer.*`, `avatar.cache_dir`.

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

On first run with a given image, the pipeline pre-processes the reference image and caches the result:

1. Detects face landmarks and bounding box via MMPose + FaceAlignment
2. Crops and resizes the face region to 256×256
3. Encodes the crop through the VAE to produce a reference latent
4. Computes blending masks via the face-parsing model
5. Saves result to `cache/avatars/<hash>.pkl` (keyed by image content + bbox params)

On subsequent startups with the same image, step 1–4 are skipped and the cache loads in ~50 ms instead of 2–5 s. Delete `cache/avatars/` to force reprocessing. **No per-frame avatar encoding** during inference — only the audio-conditioned UNet runs per batch.

---

## Project Structure

```
.
├── run_musetalk_avatar.py        # Live avatar server (entry point)
├── musetalk_avatar_pipeline.py   # Core streaming pipeline
├── tts_kokoro.py                 # Kokoro TTS wrapper
├── llm_wrapper.py                # Streaming LLM (Ollama / OpenAI / Echo)
├── config.py                     # Runtime config loader (omegaconf)
├── config.yaml                   # User-editable config (voice, chunking, enhancer…)
├── musetalk/
│   └── utils/
│       ├── audio_processor.py    # Whisper feature extraction (accepts BytesIO)
│       ├── blending.py           # Frame compositing
│       ├── enhancer.py           # GFPGAN post-processing (toggleable via config)
│       ├── face_parsing.py
│       ├── preprocessing.py
│       └── utils.py
├── cache/
│   └── avatars/                  # Pre-processed avatar cache (auto-generated)
├── models/                       # Downloaded weights
├── experiments/pretrained_models/
│   └── GFPGANv1.4.pth            # Optional GFPGAN weights (v1.3 also works)
├── scripts/inference.py
├── configs/
├── download_weights.sh
└── setup_mmlab.sh
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
