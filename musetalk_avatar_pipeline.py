"""
musetalk_avatar_pipeline.py
---------------------------
Orchestrates: User text → LLM (streaming) → Kokoro TTS → MuseTalk → synced output

Architecture mirrors live_avatar_pipeline.py from FlashHead but uses MuseTalk
for video generation, which runs efficiently on RTX 3080.

Thread model:
  ┌──────────────────┐  text_q  ┌────────────────────────────────┐  output_q  ┌────────┐
  │ LLM stream thread│ ───────► │  TTS + MuseTalk worker         │ ─────────► │ Output │
  │ (per send() call)│          │  (serial per sentence)         │            │ thread │
  └──────────────────┘          └────────────────────────────────┘            └────────┘

Key differences from FlashHead pipeline:
  - MuseTalk uses Whisper audio features (not wav2vec2)
  - Avatar is pre-processed once from a reference image/video
  - VAE + UNet only during inference (no diffusion steps) → much faster on 3080
  - Frames are blended back onto the original face region
"""

from __future__ import annotations

import io
import os
import re
import threading
import queue
import time
import tempfile
import wave
from collections import namedtuple
from typing import Iterator, Optional, List

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

TTS_SR = 16_000   # Kokoro output sample rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audio_to_wav_file(audio: np.ndarray, sr: int = TTS_SR) -> str:
    """Write float32 PCM array to a temp WAV file. Returns the file path."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return tmp.name


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class MuseTalkAvatarPipeline:
    """
    Parameters
    ----------
    unet_config      : path to musetalkV15/musetalk.json
    unet_model_path  : path to musetalkV15/unet.pth
    whisper_dir      : path to whisper model directory
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
        tts_voice: str = "af_heart",
        tts_speed: float = 1.0,
        llm_fn=None,
        output_queue_maxsize: int = 4,
    ):
        self.fps               = fps
        self.batch_size        = batch_size
        self.bbox_shift        = bbox_shift
        self.extra_margin      = extra_margin
        self.audio_padding_left  = audio_padding_left
        self.audio_padding_right = audio_padding_right
        self.tts_voice         = tts_voice
        self.tts_speed         = tts_speed
        self._avatar_image     = avatar_image

        raw_fn = llm_fn or (lambda prompt: prompt)
        self.llm_fn = self._ensure_streaming(raw_fn)

        self._text_q  : queue.Queue = queue.Queue(maxsize=32)
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

        # Whisper
        self._audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        self._whisper = WhisperModel.from_pretrained(whisper_dir)
        self._whisper = self._whisper.to(
            device=self._device, dtype=self._weight_dtype
        ).eval()
        self._whisper.requires_grad_(False)

        # Face parser
        self._fp = FaceParsing()

        logger.info("MuseTalk models loaded.")

        # ── Pre-process avatar reference image ───────────────────────────
        logger.info(f"Pre-processing avatar image: {avatar_image}")
        self._preprocess_avatar(avatar_image)
        logger.info("Avatar pre-processing complete.")

        # Worker thread
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="TTS+MuseTalk"
        )

    # ------------------------------------------------------------------
    # Avatar pre-processing  (runs once at startup)
    # ------------------------------------------------------------------

    def _preprocess_avatar(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Avatar image not found: {image_path}")

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

        self._frame_list_cycle        = frame_list + frame_list[::-1]
        self._coord_list_cycle        = coord_list + coord_list[::-1]
        self._input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self._mask_list_cycle         = mask_list + mask_list[::-1]
        self._mask_coords_list_cycle  = mask_coords_list + mask_coords_list[::-1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._running.set()
        self._worker_thread.start()
        logger.info("MuseTalkAvatarPipeline started.")

    def stop(self):
        self._running.clear()
        self._text_q.put(_STOP)
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
                logger.info(f"[LLM] sentence #{count+1}: {sentence!r}")
                self._text_q.put(sentence)
                count += 1
        except Exception as exc:
            logger.error(f"[LLM stream] error: {exc}")
        finally:
            logger.info(f"[LLM] stream done — {count} sentence(s) queued")

    # ------------------------------------------------------------------
    # Internal worker — TTS then MuseTalk
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _worker(self):
        logger.info("[Worker] TTS+MuseTalk worker started")

        while self._running.is_set():
            try:
                item = self._text_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _STOP:
                self.output_q.put(_STOP)
                break

            if self._interrupt.is_set():
                continue

            sentence = item

            # ── Step 1: TTS ──────────────────────────────────────────────
            logger.info(f"[TTS] synthesising: {sentence!r}")
            t0 = time.time()
            audio = kokoro_synthesize(sentence, voice=self.tts_voice, speed=self.tts_speed)
            logger.info(f"[TTS] done in {time.time()-t0:.2f}s  ({len(audio)/TTS_SR:.2f}s audio)")

            if self._interrupt.is_set():
                continue

            # Write audio to temp WAV for Whisper processing
            wav_path = _audio_to_wav_file(audio, TTS_SR)

            try:
                # ── Step 2: Extract Whisper audio features ───────────────
                t0 = time.time()
                whisper_input_features, librosa_length = \
                    self._audio_processor.get_audio_feature(
                        wav_path, weight_dtype=self._weight_dtype
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
                logger.info(f"[Whisper] features extracted in {time.time()-t0:.2f}s  "
                            f"({len(whisper_chunks)} chunks)")

                if self._interrupt.is_set():
                    continue

                # ── Step 3: MuseTalk UNet inference ──────────────────────
                t0 = time.time()
                res_frame_list: List[np.ndarray] = []

                gen = datagen(
                    whisper_chunks,
                    self._input_latent_list_cycle,
                    self.batch_size,
                )

                for whisper_batch, latent_batch in gen:
                    if self._interrupt.is_set():
                        break
                    audio_feature_batch = self._pe(whisper_batch.to(self._device))
                    latent_batch = latent_batch.to(
                        device=self._device, dtype=self._weight_dtype
                    )
                    pred_latents = self._unet.model(
                        latent_batch,
                        self._timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    recon = self._vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)

                logger.info(f"[UNet] {len(res_frame_list)} frames in {time.time()-t0:.2f}s")

                if self._interrupt.is_set():
                    continue

                # ── Step 4: Blend frames back onto original ───────────────
                t0 = time.time()
                blended_frames: List[np.ndarray] = []
                cycle_len = len(self._frame_list_cycle)

                for idx, res_frame in enumerate(res_frame_list):
                    ci  = idx % cycle_len
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
                    combined = enhance_frame(combined)
                    blended_frames.append(combined)

                logger.info(f"[Blend] {len(blended_frames)} frames in {time.time()-t0:.2f}s")

                # ── Step 5: Sync debug + emit ─────────────────────────────
                audio_duration_s = len(audio) / TTS_SR
                video_duration_s = len(blended_frames) / self.fps
                playback_rate    = audio_duration_s / video_duration_s if video_duration_s > 0 else 1.0

                logger.info(
                    f"[SYNC DEBUG] "
                    f"audio={audio_duration_s:.2f}s | "
                    f"actual_frames={len(blended_frames)} | "
                    f"video_duration={video_duration_s:.2f}s | "
                    f"playback_rate={playback_rate:.2f}x"
                )

                chunk = SyncedChunk(
                    audio=audio,
                    frames=blended_frames,
                    fps=self.fps,
                )
                self.output_q.put(chunk)
                logger.info(
                    f"[Worker] queued SyncedChunk: "
                    f"{audio_duration_s:.2f}s audio, {len(blended_frames)} frames"
                )

            finally:
                # Always clean up temp WAV
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        logger.info("[Worker] exited")

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