"""
run_musetalk_avatar.py
----------------------
Live Avatar Pipeline using MuseTalk — browser canvas + Web Audio, lip-sync correct.

Architecture mirrors run_live_avatar.py from FlashHead.

Server  →  /sync_feed  (SSE)  →  Browser
  • Each SyncedChunk (audio + frames) is sent as one SSE 'chunk' event
  • Browser uses AudioContext as master clock for lip-sync
  • Audio playbackRate is adjusted dynamically to match video duration

Open http://localhost:7860
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import queue
import threading
import time
import wave

import numpy as np
from flask import Flask, Response, jsonify, request
from loguru import logger

import config as cfg_module
from .pipeline import MuseTalkAvatarPipeline, SyncedChunk
from musetalk.utils.enhancer import set_enabled as set_enhancer_enabled
from .llm import build_llm

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = Flask(__name__)

_pipeline: MuseTalkAvatarPipeline = None

_system_state      = "loading"
_system_state_lock = threading.Lock()

_reset_listeners      : list = []
_reset_listeners_lock = threading.Lock()


def _broadcast_reset():
    with _reset_listeners_lock:
        for q in list(_reset_listeners):
            try:
                q.put_nowait("reset")
            except queue.Full:
                pass


def _set_state(s: str):
    global _system_state
    with _system_state_lock:
        _system_state = s
    logger.info(f"[State] -> {s}")


def _get_state() -> str:
    with _system_state_lock:
        return _system_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_wav_b64(audio: np.ndarray, sr: int = 16_000) -> str:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _encode_frame(f) -> str:
    """Encode a single BGR numpy frame to JPEG base64. Called from thread pool."""
    from PIL import Image
    import cv2
    if isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[2] == 3:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(f.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=65)
    return base64.b64encode(buf.getvalue()).decode()


def _frames_to_jpeg_b64_list(frames) -> list:
    # Encode all frames in parallel across CPU cores (~80-120ms → ~20-40ms)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, (os.cpu_count() or 4))
    ) as pool:
        return list(pool.map(_encode_frame, frames))


# ---------------------------------------------------------------------------
# HTML — dark terminal aesthetic with green accents
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Your Companion</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #FEF6EC;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: Georgia, 'Times New Roman', serif;
      color: #3D2B1F;
      padding: 24px 16px;
      gap: 20px;
    }

    h1 {
      font-size: 26px;
      font-weight: bold;
      color: #7B3F00;
      text-align: center;
      letter-spacing: 0.02em;
    }

    /* Avatar */
    #video-wrap {
      position: relative;
      width: 460px;
      height: 460px;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(123, 63, 0, 0.18);
      background: #F5E6D0;
    }

    #avatar {
      width: 460px;
      height: 460px;
      display: block;
    }

    /* Loading overlay */
    #loading-overlay {
      position: absolute;
      inset: 0;
      background: rgba(254, 246, 236, 0.93);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
      transition: opacity 0.5s;
    }
    #loading-overlay.hidden { opacity: 0; pointer-events: none; }

    .spinner {
      width: 56px; height: 56px;
      border: 5px solid #E8C99A;
      border-top-color: #C87941;
      border-radius: 50%;
      animation: spin 1.2s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    #loading-msg {
      font-size: 18px;
      color: #7B3F00;
      text-align: center;
      max-width: 260px;
      line-height: 1.5;
    }

    /* Status */
    #status-bar {
      font-size: 17px;
      color: #7B5C3A;
      background: #FDF0DC;
      border: 2px solid #E8C99A;
      border-radius: 10px;
      padding: 10px 20px;
      width: 460px;
      text-align: center;
    }

    /* Unmute */
    #unmute-banner {
      display: none;
      font-size: 17px;
      color: #7B3F00;
      background: #FFF3CD;
      border: 2px solid #F0C040;
      border-radius: 10px;
      padding: 12px 20px;
      width: 460px;
      text-align: center;
      cursor: pointer;
    }
    #unmute-banner:hover { background: #FFE99A; }

    /* Chat input */
    #chat-area {
      width: 460px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    #chat-form {
      display: flex;
      gap: 10px;
    }

    #msg {
      flex: 1;
      padding: 14px 18px;
      font-family: Georgia, serif;
      font-size: 18px;
      color: #3D2B1F;
      background: #FFFFFF;
      border: 2px solid #D4A96A;
      border-radius: 10px;
      outline: none;
      transition: border-color 0.2s;
    }
    #msg:focus        { border-color: #C87941; }
    #msg::placeholder { color: #B09070; }
    #msg:disabled     { background: #F5EDE0; color: #B09070; cursor: not-allowed; }

    #send-btn {
      padding: 14px 28px;
      font-family: Georgia, serif;
      font-size: 18px;
      font-weight: bold;
      color: #FFFFFF;
      background: #C87941;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      white-space: nowrap;
    }
    #send-btn:hover:not(:disabled) { background: #A85F2A; transform: translateY(-1px); }
    #send-btn:active:not(:disabled){ transform: translateY(0); }
    #send-btn:disabled { background: #D4B896; cursor: not-allowed; }
  </style>
</head>
<body>

  <h1>Your Companion</h1>

  <div id="video-wrap">
    <canvas id="avatar" width="460" height="460"></canvas>
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div id="loading-msg">Getting ready,<br>please wait…</div>
    </div>
  </div>

  <div id="status-bar">Starting up…</div>

  <div id="chat-area">
    <div id="unmute-banner">🔇 Tap here to turn on sound</div>
    <div id="chat-form">
      <input type="text" id="msg" placeholder="Type something here…"
             autofocus autocomplete="off" disabled />
      <button id="send-btn" disabled>Send</button>
    </div>
  </div>

<script>
const canvas       = document.getElementById('avatar');
const ctx2d        = canvas.getContext('2d');
const overlay      = document.getElementById('loading-overlay');
const loadingMsg   = document.getElementById('loading-msg');
const statusBar    = document.getElementById('status-bar');
const input        = document.getElementById('msg');
const sendBtn      = document.getElementById('send-btn');
const unmuteBanner = document.getElementById('unmute-banner');

let audioCtx   = null;
let nextPlayAt = 0;

function getCtx() {
  if (!audioCtx) {
    audioCtx   = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    nextPlayAt = audioCtx.currentTime;
  }
  return audioCtx;
}

function b64ToArrayBuffer(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return bytes.buffer.slice(0);
}

async function decodeWav(b64)  { return getCtx().decodeAudioData(b64ToArrayBuffer(b64)); }
async function decodeJpeg(b64) {
  const blob = new Blob([new Uint8Array(b64ToArrayBuffer(b64))], { type: 'image/jpeg' });
  return createImageBitmap(blob);
}

async function scheduleChunk(data) {
  const ctx = getCtx();
  if (ctx.state === 'suspended') { unmuteBanner.style.display = 'block'; return; }

  const results = await Promise.all([
    decodeWav(data.audio),
    ...data.frames.map(decodeJpeg),
  ]);
  const audioBuffer     = results[0];
  const bitmaps         = results.slice(1);
  const fps             = data.fps;

  const videoDurationS  = bitmaps.length / fps;
  const audioDurationS  = audioBuffer.duration;
  const playbackRate    = Math.min(1.0, audioDurationS / videoDurationS);
  const frameDurationMs = (videoDurationS * 1000) / bitmaps.length;

  const now = ctx.currentTime;
  if (nextPlayAt < now + 0.01) nextPlayAt = now + 0.01;

  const startAt  = nextPlayAt;
  nextPlayAt    += videoDurationS;

  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.playbackRate.value = playbackRate;
  src.connect(ctx.destination);
  src.start(startAt);

  const baseMs = (startAt - ctx.currentTime) * 1000;
  bitmaps.forEach((bm, i) => {
    const delay  = baseMs + i * frameDurationMs;
    const isLast = i === bitmaps.length - 1;
    setTimeout(() => {
      ctx2d.drawImage(bm, 0, 0, canvas.width, canvas.height);
      if (!isLast && bm.close) bm.close();
    }, Math.max(0, delay));
  });
}

function connectSyncFeed() {
  const es = new EventSource('/sync_feed');
  es.addEventListener('chunk', async (e) => {
    try { await scheduleChunk(JSON.parse(e.data)); }
    catch (err) { console.error('[sync] chunk error:', err); }
  });
  es.addEventListener('reset', () => {
    if (audioCtx) nextPlayAt = audioCtx.currentTime + 0.05;
  });
  es.onerror = () => { es.close(); setTimeout(connectSyncFeed, 2000); };
}
connectSyncFeed();

unmuteBanner.addEventListener('click', async () => {
  const ctx = getCtx();
  await ctx.resume();
  unmuteBanner.style.display = 'none';
  nextPlayAt = ctx.currentTime + 0.05;
});

const STATE_MSG = {
  loading: 'Getting ready, please wait…',
  warming: 'Almost there…',
  ready:   'Ready — type a message below',
};
let isReady = false;
async function pollStatus() {
  try {
    const d = await (await fetch('/status')).json();
    loadingMsg.innerHTML  = (STATE_MSG[d.state] || d.state).replace(',', ',<br>');
    statusBar.textContent = STATE_MSG[d.state] || d.state;
    if (d.state === 'ready' && !isReady) {
      isReady = true;
      overlay.classList.add('hidden');
      input.disabled = sendBtn.disabled = false;
      input.focus();
    }
    if (!isReady) setTimeout(pollStatus, 1500);
  } catch { setTimeout(pollStatus, 2000); }
}
pollStatus();

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });

async function sendMessage() {
  const text = input.value.trim();
  if (!text || !isReady) return;
  input.value = '';
  input.disabled = sendBtn.disabled = true;
  statusBar.textContent = 'Responding…';
  try {
    await fetch('/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  } finally {
    input.disabled = sendBtn.disabled = false;
    statusBar.textContent = 'Ready — type a message below';
    input.focus();
  }
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/status")
def status():
    return jsonify({"state": _get_state()})


@app.route("/config", methods=["GET", "POST"])
def config_endpoint():
    """
    GET  → current runtime config as JSON
    POST → deep-merge JSON body into config; applies live changes where possible
           (TTS voice/speed, enhancer on/off, chunking params).
           UNet/avatar/server changes require restart.
    """
    if request.method == "GET":
        return jsonify(cfg_module.to_dict())

    patch = request.get_json(silent=True) or {}
    new_cfg = cfg_module.patch(patch)

    # Apply live changes
    if _pipeline is not None:
        tts = patch.get("tts") or {}
        if "voice" in tts:    _pipeline.tts_voice    = tts["voice"]
        if "speed" in tts:    _pipeline.tts_speed    = float(tts["speed"])
        if "language" in tts: _pipeline.tts_language = tts["language"]

        ch = patch.get("chunking") or {}
        if "enabled"   in ch: _pipeline.chunking_enabled = bool(ch["enabled"])
        if "min_chars" in ch: _pipeline.chunk_min_chars  = int(ch["min_chars"])

        en = patch.get("enhancer") or {}
        if "enabled" in en:
            set_enhancer_enabled(bool(en["enabled"]))

    return jsonify(cfg_module.to_dict())


@app.route("/send", methods=["POST"])
def send_message():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"status": "empty input"})
    if _pipeline is None or _get_state() != "ready":
        return jsonify({"status": "still warming up, please wait"})
    _broadcast_reset()
    threading.Thread(target=_pipeline.send, args=(text,), daemon=True).start()
    return jsonify({"status": f"Generating: {text[:50]}…"})


@app.route("/sync_feed")
def sync_feed():
    client_reset_q: queue.Queue = queue.Queue(maxsize=4)
    with _reset_listeners_lock:
        _reset_listeners.append(client_reset_q)

    def generate():
        chunk_q: queue.Queue = queue.Queue(maxsize=4)

        def _pusher():
            if _pipeline is None:
                chunk_q.put(None)
                return
            for chunk in _pipeline.iter_chunks(timeout=7200):
                chunk_q.put(chunk)
            chunk_q.put(None)

        threading.Thread(target=_pusher, daemon=True).start()

        try:
            while True:
                try:
                    sig = client_reset_q.get_nowait()
                    if sig == "reset":
                        yield "event: reset\ndata: {}\n\n"
                except queue.Empty:
                    pass

                try:
                    item = chunk_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                if not isinstance(item, SyncedChunk):
                    continue

                payload = json.dumps({
                    "audio":  _numpy_to_wav_b64(item.audio, sr=16_000),
                    "frames": _frames_to_jpeg_b64_list(item.frames),
                    "fps":    item.fps,
                })
                yield f"event: chunk\ndata: {payload}\n\n"
        finally:
            with _reset_listeners_lock:
                try:
                    _reset_listeners.remove(client_reset_q)
                except ValueError:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Args + Main
# ---------------------------------------------------------------------------

def parse_args():
    """CLI args. Only --avatar_image is required; everything else falls back
    to config.yaml. Pass an explicit value to override the config."""
    p = argparse.ArgumentParser(description="Live Avatar — MuseTalk + Kokoro TTS")
    p.add_argument("--avatar_image",    required=True, help="Path to reference face image")
    p.add_argument("--unet_config",     default="./models/musetalkV15/musetalk.json")
    p.add_argument("--unet_model_path", default="./models/musetalkV15/unet.pth")
    p.add_argument("--whisper_dir",     default="./models/whisper")
    p.add_argument("--vae_type",        default="sd-vae")
    # Config overrides (None = use config.yaml value)
    p.add_argument("--tts_voice",       default=None,  help="Override config.tts.voice")
    p.add_argument("--tts_speed",       default=None, type=float, help="Override config.tts.speed")
    p.add_argument("--batch_size",      default=None, type=int)
    p.add_argument("--bbox_shift",      default=None, type=int)
    p.add_argument("--extra_margin",    default=None, type=int)
    p.add_argument("--fps",             default=None, type=int)
    p.add_argument("--port",            default=None, type=int)
    # LLM backend
    p.add_argument("--llm_backend",     default="echo", choices=["echo", "openai", "ollama"])
    p.add_argument("--llm_model",       default="llama3.2")
    p.add_argument("--llm_api_key",     default=None)
    p.add_argument("--llm_base_url",    default=None)
    p.add_argument("--ollama_host",     default="http://localhost:11434")
    return p.parse_args()


def main():
    global _pipeline
    args = parse_args()
    cfg = cfg_module.get()

    # CLI overrides over config.yaml
    tts_voice    = args.tts_voice    if args.tts_voice    is not None else cfg.tts.voice
    tts_speed    = args.tts_speed    if args.tts_speed    is not None else cfg.tts.speed
    batch_size   = args.batch_size   if args.batch_size   is not None else cfg.unet.batch_size
    bbox_shift   = args.bbox_shift   if args.bbox_shift   is not None else cfg.avatar.bbox_shift
    extra_margin = args.extra_margin if args.extra_margin is not None else cfg.avatar.extra_margin
    fps          = args.fps          if args.fps          is not None else cfg.unet.fps
    port         = args.port         if args.port         is not None else cfg.server.port

    set_enhancer_enabled(cfg.enhancer.enabled)

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def _init_and_warmup():
        global _pipeline
        _set_state("loading")

        llm_fn = build_llm(
            backend=args.llm_backend,
            model=args.llm_model,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            ollama_host=args.ollama_host,
        )

        _pipeline = MuseTalkAvatarPipeline(
            unet_config=args.unet_config,
            unet_model_path=args.unet_model_path,
            whisper_dir=args.whisper_dir,
            avatar_image=args.avatar_image,
            vae_type=args.vae_type,
            use_float16=cfg.unet.use_float16,
            batch_size=batch_size,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            fps=fps,
            audio_padding_left=cfg.unet.audio_padding_left,
            audio_padding_right=cfg.unet.audio_padding_right,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            tts_language=cfg.tts.language,
            llm_fn=llm_fn,
            compile_unet=cfg.unet.compile,
            chunking_enabled=cfg.chunking.enabled,
            chunk_min_chars=cfg.chunking.min_chars,
            avatar_cache_dir=cfg.avatar.cache_dir,
        )

        _set_state("warming")
        _pipeline.start()
        _set_state("ready")
        logger.info("[Init] Pipeline ready for input.")

    threading.Thread(target=_init_and_warmup, daemon=True, name="Init").start()

    print("\n" + "=" * 60)
    print(f"  Avatar stream ->  http://localhost:{port}")
    print(f"  Voice: {tts_voice}  |  Chunking: {cfg.chunking.enabled}  |  Enhancer: {cfg.enhancer.enabled}")
    print("  Ctrl-C to stop")
    print("=" * 60 + "\n")

    try:
        app.run(host=cfg.server.host, port=port, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        if _pipeline:
            _pipeline.stop()
        print("Done.")


if __name__ == "__main__":
    main()
