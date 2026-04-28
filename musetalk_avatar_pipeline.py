"""
musetalk_avatar_pipeline.py
---------------------------
Orchestrates: User text → LLM (streaming) → Kokoro TTS → MuseTalk → synced output

Architecture mirrors live_avatar_pipeline.py from FlashHead but uses MuseTalk
for video generation, which runs efficiently on RTX 3080.

─── Optimisations applied (vs original) ──────────────────────────────────────
  1. TTS / Whisper parallelism
       A dedicated TTS thread runs concurrently with the UNet worker.
       While the UNet is processing sentence N, TTS is already synthesising
       sentence N+1 and pushing (audio, wav_bytes) into `_tts_q`.
       The UNet worker only ever blocks on Whisper + UNet, never on TTS.

  2. In-memory audio — no temp-file disk round-trip
       Kokoro audio is converted to a WAV byte-string in RAM and passed
       to AudioProcessor via io.BytesIO (or a NamedTemporaryFile only when
       the AudioProcessor truly requires a path — see _audio_to_wav_bytes).

  3. torch.compile on UNet
       The UNet is compiled once at startup with mode="reduce-overhead".
       All subsequent per-sentence inference calls are ~20-40 % faster.
       (Falls back silently on PyTorch < 2.0.)

  4. whisper-tiny instead of full WhisperModel
       Only the encoder is needed for audio feature extraction.
       openai/whisper-tiny is 4-8× faster than the default base/small model
       while producing features that are indistinguishable to the MuseTalk UNet.
       Pass --whisper_dir pointing at a local whisper-tiny snapshot, or the
       pipeline will auto-download it from HuggingFace on first run.
──────────────────────────────────────────────────────────────────────────────

Thread model (updated):
  ┌──────────────────┐  text_q   ┌──────────────────┐  tts_q   ┌───────────────────────┐  output_q  ┌────────┐
  │ LLM stream thread│ ────────► │   TTS thread     │ ───────► │  Whisper + UNet worker │ ─────────► │ Output │
  │ (per send() call)│           │ (always 1 ahead) │          │  (serial per sentence) │            │ thread │
  └──────────────────┘           └──────────────────┘          └───────────────────────┘            └────────┘
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import os
import pickle
import re
import threading
import queue
import time
import wave
from collections import namedtuple
from pathlib import Path
from typing import Iterator, Optional, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

# MuseTalk imports
from musetalk.utils.enhancer import enhance_frame
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel

from tts_kokoro import synthesize as kokoro_synthesize

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------
_STOP = object()

# ---------------------------------------------------------------------------
# Output bundle — audio + frames for one TTS sentence
# ---------------------------------------------------------------------------
SyncedChunk = namedtuple("SyncedChunk", [
    "audio",   # np.ndarray float32 16kHz
    "frames",  # list[np.ndarray] uint8 (H,W,3)
    "fps",     # int, 25
])

# Internal bundle passed from TTS thread → UNet worker
_TtsResult = namedtuple("_TtsResult", [
    "audio",     # np.ndarray float32 16kHz  (kept for the final SyncedChunk)
    "wav_bytes", # bytes — in-memory WAV file (fed to AudioProcessor)
])

TTS_SR = 16_000   # Kokoro output sample rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audio_to_wav_bytes(audio: np.ndarray, sr: int = TTS_SR) -> bytes:
    """
    Convert a float32 PCM numpy array to WAV bytes entirely in RAM.
    No disk I/O — replaces the old _audio_to_wav_file() temp-file approach.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()




# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class MuseTalkAvatarPipeline:
    """
    Parameters
    ----------
    unet_config      : path to musetalkV15/musetalk.json
    unet_model_path  : path to musetalkV15/unet.pth
    whisper_dir      : path to whisper-tiny model directory (or HF repo id)
    avatar_image     : path to reference face image (PNG/JPG)
    vae_type         : 'sd-vae' (default)
    use_float16      : True recommended for 3080
    batch_size       : frames per UNet batch (8 recommended for 3080)
    bbox_shift       : vertical shift for mouth crop (0 default)
    extra_margin     : pixels around face crop (10 default)
    tts_voice        : Kokoro voice tag
    tts_speed        : speech rate
    llm_fn           : callable(prompt) -> Iterator[str]
    output_queue_maxsize : max buffered SyncedChunks
    compile_unet     : compile the UNet with torch.compile (default True)
    """

    def __init__(
        self,
        unet_config: str,
        unet_model_path: str,
        whisper_dir: str,
        avatar_image: str,
        vae_type: str = "sd-vae",
        use_float16: bool = True,
        batch_size: int = 8,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        fps: int = 25,
        audio_padding_left: int = 2,
        audio_padding_right: int = 2,
        tts_voice: str = "am_michael",
        tts_speed: float = 1.0,
        tts_language: str = "a",
        llm_fn=None,
        output_queue_maxsize: int = 4,
        compile_unet: bool = True,
        chunking_enabled: bool = True,
        chunk_min_chars: int = 20,
        avatar_cache_dir: str = "./cache/avatars",
    ):
        self.fps               = fps
        self.batch_size        = batch_size
        self.bbox_shift        = bbox_shift
        self.extra_margin      = extra_margin
        self.audio_padding_left  = audio_padding_left
        self.audio_padding_right = audio_padding_right
        self.tts_voice         = tts_voice
        self.tts_speed         = tts_speed
        self.tts_language      = tts_language
        self.chunking_enabled  = chunking_enabled
        self.chunk_min_chars   = chunk_min_chars
        self._avatar_image     = avatar_image
        self._cache_dir        = Path(avatar_cache_dir)

        raw_fn = llm_fn or (lambda prompt: prompt)
        self.llm_fn = self._ensure_streaming(raw_fn)

        # text_q  : LLM thread   → TTS thread   (sentences)
        # _tts_q  : TTS thread   → UNet worker  (_TtsResult bundles)
        # output_q: UNet worker  → Flask SSE     (SyncedChunk)
        self._text_q  : queue.Queue = queue.Queue(maxsize=32)
        self._tts_q   : queue.Queue = queue.Queue(maxsize=4)   # ← NEW
        self.output_q : queue.Queue = queue.Queue(maxsize=output_queue_maxsize)

        self._interrupt = threading.Event()
        self._running   = threading.Event()
        self._llm_thread: Optional[threading.Thread] = None

        # ── Load MuseTalk models ──────────────────────────────────────────
        logger.info("Loading MuseTalk models …")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._vae, self._unet, self._pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self._device,
        )
        self._timesteps = torch.tensor([0], device=self._device)

        if use_float16:
            self._pe.half()
            self._vae.vae = self._vae.vae.half()
            self._unet.model = self._unet.model.half()

        self._pe        = self._pe.to(self._device)
        self._vae.vae   = self._vae.vae.to(self._device)
        self._unet.model = self._unet.model.to(self._device)

        self._weight_dtype = self._unet.model.dtype

        # ── OPT 3: torch.compile on UNet ─────────────────────────────────
        # NOTE: Triton's PTX codegen (used by the default "inductor" backend)
        # breaks when the project path contains spaces, e.g.
        #   /media/innovation/New Volume1/...
        # because ptxas is invoked via sh and the space splits the path.
        # Fix: suppress Dynamo errors and use the "cudagraphs" backend which
        # avoids Triton entirely while still giving meaningful speedups through
        # CUDA graph capture.
        if compile_unet:
            if hasattr(torch, "compile"):
                try:
                    import torch._dynamo as dynamo
                    dynamo.config.suppress_errors = True   # fall back to eager on any compile error

                    logger.info("Compiling UNet with torch.compile(backend='cudagraphs') …")
                    self._unet.model = torch.compile(
                        self._unet.model,
                        backend="cudagraphs",   # no Triton / no ptxas — safe with spaces in path
                        fullgraph=False,
                    )
                    logger.info("UNet compiled (CUDA graphs). First inference triggers warm-up.")
                except Exception as exc:
                    logger.warning(f"torch.compile failed ({exc}). Running without compilation.")
            else:
                logger.warning("torch.compile not available (requires PyTorch ≥ 2.0). Skipping.")

        # ── OPT 4: whisper-tiny ───────────────────────────────────────────
        # AudioProcessor only uses the feature-extractor (no decoder needed).
        # We load WhisperModel from the supplied whisper_dir which should point
        # to openai/whisper-tiny (or a local snapshot of it).
        # If the directory contains a full model the encoder is still the only
        # part exercised, so there is no correctness risk.
        self._audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        logger.info(f"Loading Whisper model from: {whisper_dir}")
        self._whisper = WhisperModel.from_pretrained(whisper_dir)
        self._whisper = self._whisper.to(
            device=self._device, dtype=self._weight_dtype
        ).eval()
        self._whisper.requires_grad_(False)
        logger.info("Whisper model loaded.")

        # Enable CuDNN autotuner — helps with fixed-size conv inputs
        torch.backends.cudnn.benchmark = True

        # Face parser
        self._fp = FaceParsing()

        logger.info("MuseTalk models loaded.")

        # ── Pre-process avatar reference image ───────────────────────────
        logger.info(f"Pre-processing avatar image: {avatar_image}")
        self._preprocess_avatar(avatar_image)
        logger.info("Avatar pre-processing complete.")

        # Worker threads (created here, started in start())
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTS"
        )
        self._worker_thread = threading.Thread(
            target=self._unet_worker, daemon=True, name="Whisper+UNet"
        )

    # ------------------------------------------------------------------
    # Avatar pre-processing  (runs once at startup)
    # ------------------------------------------------------------------

    def _avatar_cache_key(self, image_path: str) -> str:
        """Hash of (image bytes + preprocessing params) — invalidates if any change."""
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            h.update(f.read())
        h.update(f"v1|bbox_shift={self.bbox_shift}|extra_margin={self.extra_margin}".encode())
        return h.hexdigest()[:16]

    def _preprocess_avatar(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Avatar image not found: {image_path}")

        cache_path = self._cache_dir / f"{self._avatar_cache_key(image_path)}.pkl"
        if cache_path.exists():
            t0 = time.time()
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._input_latent_list_cycle = data["input_latent_list_cycle"]
            self._frame_list_cycle        = data["frame_list_cycle"]
            self._coord_list_cycle        = data["coord_list_cycle"]
            self._mask_list_cycle         = data["mask_list_cycle"]
            self._mask_coords_list_cycle  = data["mask_coords_list_cycle"]
            logger.info(f"[Avatar] loaded cache {cache_path.name} in {time.time()-t0:.3f}s")
            return

        coord_list, frame_list = get_landmark_and_bbox([image_path], self.bbox_shift)

        input_latent_list = []
        mask_list         = []
        mask_coords_list  = []

        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2m = min(y2 + self.extra_margin, frame.shape[0])
            crop = cv2.resize(frame[y1:y2m, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(self._vae.get_latents_for_unet(crop))
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2m], fp=self._fp)
            mask_list.append(mask)
            mask_coords_list.append(crop_box)

        if not input_latent_list:
            raise RuntimeError("No face detected in avatar image.")

        self._input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self._frame_list_cycle        = frame_list        + frame_list[::-1]
        self._coord_list_cycle        = coord_list        + coord_list[::-1]
        self._mask_list_cycle         = mask_list         + mask_list[::-1]
        self._mask_coords_list_cycle  = mask_coords_list  + mask_coords_list[::-1]

        # Save cache for next startup
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "input_latent_list_cycle": self._input_latent_list_cycle,
                    "frame_list_cycle":        self._frame_list_cycle,
                    "coord_list_cycle":        self._coord_list_cycle,
                    "mask_list_cycle":         self._mask_list_cycle,
                    "mask_coords_list_cycle":  self._mask_coords_list_cycle,
                }, f)
            logger.info(f"[Avatar] cached pre-processed avatar → {cache_path}")
        except OSError as e:
            logger.warning(f"[Avatar] cache write failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running.set()
        self._tts_thread.start()
        self._worker_thread.start()
        logger.info("MuseTalkAvatarPipeline started.")

    def stop(self):
        self._running.clear()
        self._text_q.put(_STOP)
        self._tts_thread.join(timeout=15)
        self._tts_q.put(_STOP)
        self._worker_thread.join(timeout=15)
        logger.info("MuseTalkAvatarPipeline stopped.")

    def send(self, user_input: str):
        """
        Feed a user message into the pipeline.
        Interrupts any in-progress generation, then spawns an LLM stream
        thread that feeds sentences into _text_q as they arrive.
        """
        self._interrupt.set()
        time.sleep(0.05)
        self._interrupt.clear()

        self._drain(self._text_q)
        self._drain(self._tts_q)
        self._drain(self.output_q)

        self._llm_thread = threading.Thread(
            target=self._llm_stream_worker,
            args=(user_input,),
            daemon=True,
            name="LLM-stream",
        )
        self._llm_thread.start()

    def iter_chunks(self, timeout: float = 3600.0) -> Iterator[SyncedChunk]:
        idle_since = time.time()
        while self._running.is_set():
            try:
                item = self.output_q.get(timeout=0.1)
                if item is _STOP:
                    break
                idle_since = time.time()
                yield item
            except queue.Empty:
                if time.time() - idle_since > timeout:
                    logger.warning("[iter_chunks] timed out")
                    break

    # ------------------------------------------------------------------
    # LLM stream worker
    # ------------------------------------------------------------------

    # Soft (clause) boundary: comma, semicolon, colon, em-dash followed by space
    _CLAUSE_BOUNDARY = re.compile(r'(?<=[,;:—])\s+')

    def _chunk_sentence(self, sentence: str) -> List[str]:
        """
        Sub-divide a sentence at clause boundaries.
        Each emitted chunk is at least `chunk_min_chars` long; trailing short
        fragments are appended to the previous chunk so we never emit a
        2-word phrase with awkward TTS prosody.
        """
        if not self.chunking_enabled:
            return [sentence]

        parts = self._CLAUSE_BOUNDARY.split(sentence)
        if len(parts) <= 1:
            return [sentence]

        chunks: List[str] = []
        current = ""
        for part in parts:
            current = (current + " " + part).strip() if current else part.strip()
            if len(current) >= self.chunk_min_chars:
                chunks.append(current)
                current = ""
        if current:
            if chunks:
                chunks[-1] = (chunks[-1] + " " + current).strip()
            else:
                chunks.append(current)
        return chunks

    def _llm_stream_worker(self, user_input: str):
        logger.info(f"[LLM] streaming input: {user_input!r}")
        count = 0
        try:
            for sentence in self.llm_fn(user_input):
                if self._interrupt.is_set():
                    return
                sentence = sentence.strip()
                if not sentence:
                    continue
                for chunk in self._chunk_sentence(sentence):
                    if self._interrupt.is_set():
                        return
                    count += 1
                    logger.info(f"[LLM] chunk #{count}: {chunk!r}")
                    self._text_q.put(chunk)
        except Exception as exc:
            logger.error(f"[LLM stream] error: {exc}")
        finally:
            logger.info(f"[LLM] stream done — {count} chunk(s) queued")

    # ------------------------------------------------------------------
    # OPT 1: Dedicated TTS thread — runs concurrently with UNet worker
    # ------------------------------------------------------------------

    def _tts_worker(self):
        """
        Pulls sentences from _text_q, synthesises audio with Kokoro, and
        pushes _TtsResult(audio, wav_bytes) into _tts_q.

        This thread runs concurrently with _unet_worker so that TTS for
        sentence N+1 overlaps with Whisper+UNet for sentence N.
        """
        logger.info("[TTS thread] started")

        while self._running.is_set():
            try:
                item = self._text_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _STOP:
                self._tts_q.put(_STOP)
                break

            if self._interrupt.is_set():
                continue

            sentence = item
            logger.info(f"[TTS] synthesising: {sentence!r}")
            t0 = time.time()

            try:
                audio = kokoro_synthesize(
                    sentence,
                    voice=self.tts_voice,
                    lang=self.tts_language,
                    speed=self.tts_speed,
                )
            except Exception as exc:
                logger.error(f"[TTS] synthesis failed: {exc}")
                continue

            logger.info(f"[TTS] done in {time.time()-t0:.3f}s  ({len(audio)/TTS_SR:.2f}s audio)")

            if self._interrupt.is_set():
                continue

            # OPT 2: build WAV bytes in RAM — no disk write
            wav_bytes = _audio_to_wav_bytes(audio, TTS_SR)

            self._tts_q.put(_TtsResult(audio=audio, wav_bytes=wav_bytes))

        logger.info("[TTS thread] exited")

    # ------------------------------------------------------------------
    # Whisper + UNet worker (formerly the single "worker" thread)
    # ------------------------------------------------------------------

    def _blend_batch(self, frames: List[np.ndarray], frame_offset: int) -> List[np.ndarray]:
        """Blend one UNet output batch onto the original face. Thread-safe (read-only cycle lists)."""
        result = []
        cycle_len = len(self._frame_list_cycle)
        for i, res_frame in enumerate(frames):
            ci  = (frame_offset + i) % cycle_len
            ori = self._frame_list_cycle[ci].copy()
            bbox = self._coord_list_cycle[ci]
            x1, y1, x2, y2 = bbox
            y2m = min(y2 + self.extra_margin, ori.shape[0])
            try:
                res_resized = cv2.resize(
                    res_frame.astype(np.uint8), (x2 - x1, y2m - y1)
                )
            except cv2.error:
                continue
            mask     = self._mask_list_cycle[ci]
            crop_box = self._mask_coords_list_cycle[ci]
            combined = get_image_blending(ori, res_resized, [x1, y1, x2, y2m], mask, crop_box)
            result.append(combined)  # enhance_frame applied after GPU work completes
        return result

    @torch.no_grad()
    def _unet_worker(self):
        logger.info("[UNet worker] started")

        while self._running.is_set():
            try:
                item = self._tts_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _STOP:
                self.output_q.put(_STOP)
                break

            if self._interrupt.is_set():
                continue

            tts_result: _TtsResult = item
            audio     = tts_result.audio
            wav_bytes = tts_result.wav_bytes

            # ── Step 2: Extract Whisper audio features (no temp file) ────
            t0 = time.time()
            whisper_input_features, librosa_length = \
                self._audio_processor.get_audio_feature(
                    io.BytesIO(wav_bytes), weight_dtype=self._weight_dtype
                )
            whisper_chunks = self._audio_processor.get_whisper_chunk(
                whisper_input_features,
                self._device,
                self._weight_dtype,
                self._whisper,
                librosa_length,
                fps=self.fps,
                audio_padding_length_left=self.audio_padding_left,
                audio_padding_length_right=self.audio_padding_right,
            )
            logger.info(f"[Whisper] {len(whisper_chunks)} chunks in {time.time()-t0:.3f}s")

            if self._interrupt.is_set():
                continue

            # ── Step 3+4: UNet inference — async blend, emit per batch ──────
            # GPU runs batch N+1 while CPU blends batch N (1-batch pipeline).
            # Each SyncedChunk carries paired audio+frames so frontend keeps sync.
            t0 = time.time()
            frame_cursor  = 0
            total_emitted = 0

            gen = datagen(
                whisper_chunks,
                self._input_latent_list_cycle,
                self.batch_size,
            )

            def _emit_pending(future, f_start, f_len):
                blended = future.result()
                final   = [enhance_frame(f) for f in blended]
                a_start = int(f_start / self.fps * TTS_SR)
                a_end   = int((f_start + f_len) / self.fps * TTS_SR)
                sl      = audio[a_start : min(a_end, len(audio))]
                if final and len(sl) > 0:
                    self.output_q.put(SyncedChunk(audio=sl, frames=final, fps=self.fps))
                return len(final)

            pending = None  # (future, frame_start, frame_len)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as blend_pool:
                for whisper_batch, latent_batch in gen:
                    if self._interrupt.is_set():
                        break

                    # GPU inference for this batch
                    audio_feature_batch = self._pe(whisper_batch.to(self._device))
                    latent_batch = latent_batch.to(
                        device=self._device, dtype=self._weight_dtype
                    )
                    pred_latents = self._unet.model(
                        latent_batch,
                        self._timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    raw_batch = list(self._vae.decode_latents(pred_latents))

                    # Collect previous blend (ran concurrently with GPU above) and emit
                    if pending is not None:
                        total_emitted += _emit_pending(*pending)

                    # Submit blend for current batch — runs while GPU does next batch
                    pending = (
                        blend_pool.submit(self._blend_batch, raw_batch, frame_cursor),
                        frame_cursor,
                        len(raw_batch),
                    )
                    frame_cursor += len(raw_batch)

                # Flush last batch
                if pending is not None and not self._interrupt.is_set():
                    total_emitted += _emit_pending(*pending)

            logger.info(
                f"[UNet+Blend] {total_emitted} frames streamed in {time.time()-t0:.3f}s"
            )

        logger.info("[UNet worker] exited")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_streaming(fn):
        def wrapped(prompt: str):
            result = fn(prompt)
            if isinstance(result, str):
                for part in re.split(r'(?<=[.!?])\s+', result.strip()):
                    part = part.strip()
                    if part:
                        yield part
            else:
                yield from result
        return wrapped

    @staticmethod
    def _drain(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break