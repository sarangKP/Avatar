# Live Avatar

A real-time conversational AI avatar that streams lip-synced video in a browser. Built on [MuseTalk](https://github.com/TMElyralab/MuseTalk) with Kokoro TTS and a streaming pipeline designed for low latency on consumer GPUs (tested on RTX 3080).

---

## How it works

```
User input → LLM → Kokoro TTS → Whisper features → MuseTalk UNet → browser canvas
```

Each sentence is processed in parallel — TTS for sentence N+1 runs while MuseTalk renders sentence N. Frames are streamed per GPU batch with paired audio slices so the browser always has a matched audio+video chunk to schedule.

---

## Project structure

```
Avatar/
├── avatar/                  # Live avatar package
│   ├── pipeline.py          # Streaming pipeline (LLM → TTS → Whisper → UNet)
│   ├── server.py            # Flask server + SSE endpoint + browser UI
│   ├── tts.py               # Kokoro TTS wrapper (24kHz → 16kHz)
│   └── llm.py               # LLM streaming backends (Ollama / OpenAI / Echo)
├── musetalk/                # MuseTalk model library
│   ├── models/              # UNet, VAE
│   ├── utils/               # Face detection, blending, audio processing, enhancer
│   └── whisper/             # Whisper audio feature extraction
├── examples/                # Sample avatar images
├── assets/                  # Documentation images
├── config.py                # Config loader (reads config.yaml, supports live patching)
├── config.yaml              # Runtime configuration
├── run.py                   # Entry point
├── requirements.txt
├── download_weights.sh      # Download model weights (Linux/Mac)
├── download_weights.bat     # Download model weights (Windows)
└── setup_mmlab.sh           # Install MMPose / DWPose dependencies
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (RTX 3080 or better recommended)
- `espeak-ng` for Kokoro TTS phonemisation:
  ```bash
  sudo apt-get install espeak-ng
  ```

---

## Installation

```bash
git clone https://github.com/sarangKP/Avatar.git
cd Avatar

# Install dependencies (uv recommended)
pip install uv
uv sync

# Or with pip
pip install -r requirements.txt

# Install MMPose for face detection
bash setup_mmlab.sh

# Download model weights (~3 GB)
bash download_weights.sh
```

---

## Running

```bash
python run.py --avatar_image examples/face_1.png --llm_backend echo
```

Open **http://localhost:7860** in your browser.

### LLM backends

| Backend | Flag | Notes |
|---------|------|-------|
| Echo (testing) | `--llm_backend echo` | Repeats your input as speech, no LLM needed |
| Ollama (local) | `--llm_backend ollama --llm_model llama3.2` | Requires [Ollama](https://ollama.com) running locally |
| OpenAI | `--llm_backend openai --llm_model gpt-4o` | Requires `OPENAI_API_KEY` env var |

### All CLI options

```bash
python run.py \
  --avatar_image examples/face_1.png \
  --llm_backend ollama \
  --llm_model llama3.2 \
  --port 7860 \
  --tts_voice am_michael \
  --tts_speed 1.0
```

---

## Configuration

Edit `config.yaml` to change settings. TTS voice/speed, chunking, and enhancer update live via `POST /config` without a restart.

```yaml
tts:
  voice: am_michael     # see voice table below
  speed: 1.0
  language: a           # 'a' = American English, 'b' = British English

enhancer:
  enabled: false        # GFPGAN face restoration — adds ~0.26s/frame, disable for speed

chunking:
  enabled: true
  min_chars: 5          # minimum chars before a clause boundary flushes to TTS

unet:
  batch_size: 4         # frames per GPU batch — lower = faster first frame
  fps: 25
  use_float16: true
  compile: false        # torch.compile — disable for consistent latency

avatar:
  bbox_shift: 0         # vertical shift for mouth crop
  extra_margin: 10      # pixels added below face bbox

server:
  host: 0.0.0.0
  port: 7860
```

### TTS voices

| Code | Description |
|------|-------------|
| `am_michael` | American English male |
| `am_adam` | American English male (alternative) |
| `bm_george` | British English male |
| `bm_lewis` | British English male (alternative) |
| `af_heart` | American English female |
| `af_sky` | American English female (alternative) |
| `bf_emma` | British English female |
| `bf_isabella` | British English female (alternative) |

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser UI |
| `GET` | `/status` | Pipeline state: `loading` / `warming` / `ready` |
| `POST` | `/send` | Send a message `{"text": "..."}` |
| `GET` | `/sync_feed` | SSE stream of audio+frame chunks |
| `GET` | `/config` | Current config as JSON |
| `POST` | `/config` | Live-patch config e.g. `{"tts": {"voice": "af_heart"}}` |

---

## Latency

Measured on RTX 3080, `compile: false`, `enhancer: false`:

| Stage | Time |
|-------|------|
| TTS synthesis | ~0.4–0.9s per sentence |
| Whisper feature extraction | ~0.01–0.7s |
| First UNet batch (4 frames) | ~50–200ms |
| **Time to first frame** | **~1–2s** |

Enable `enhancer: true` for sharper mouth rendering at the cost of ~0.26s per frame.

---

## Troubleshooting

**Slow first load** — model weights are loaded into GPU memory on startup (~3–5s). Avatar preprocessing (face detection + VAE encoding) is cached to `./cache/avatars/` after the first run.

**No audio** — click "Tap here to turn on sound". Browsers block autoplay until a user gesture.

**`espeak-ng` not found** — required by Kokoro TTS. Install with `sudo apt-get install espeak-ng`.

**CUDA out of memory** — lower `batch_size` in `config.yaml` (try `2`) or ensure `use_float16: true`.
