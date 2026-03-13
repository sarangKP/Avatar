"""
tts_kokoro.py
-------------
Kokoro TTS wrapper for the live avatar pipeline.

Kokoro outputs 24kHz audio — this module resamples to 16kHz mono
which is what wav2vec2 expects inside FlashHeadPipeline.

Install:
    pip install kokoro>=0.9.2 soundfile scipy
    # Kokoro also needs espeak-ng for phonemisation:
    # Ubuntu: apt-get install espeak-ng
"""

import re
import numpy as np
import scipy.signal as sps

# ---------------------------------------------------------------------------
# Lazy import so the rest of the pipeline can be imported without Kokoro
# ---------------------------------------------------------------------------
_kokoro_pipeline = None

TARGET_SR = 16_000          # wav2vec2 requirement
KOKORO_SR  = 24_000          # Kokoro native sample rate


def _get_kokoro(lang: str = "a", voice: str = "af_heart"):
    """Lazily initialise the Kokoro pipeline (once per process)."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code=lang)
        print(f"[TTS] Kokoro pipeline loaded  (voice={voice})")
    return _kokoro_pipeline


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 array from orig_sr → target_sr."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = sps.resample(audio, num_samples).astype(np.float32)
    return resampled


def synthesize(text: str,
               voice: str = "af_heart",
               lang: str = "a",
               speed: float = 1.0) -> np.ndarray:
    """
    Convert *text* to a 16 kHz mono float32 PCM numpy array.

    Parameters
    ----------
    text  : str   – the sentence to synthesise
    voice : str   – Kokoro voice tag (default 'af_heart', American English female)
    lang  : str   – language code ('a' = American English)
    speed : float – speech rate multiplier

    Returns
    -------
    np.ndarray, shape (N,), dtype float32, sample_rate=16000
    """
    kpipe = _get_kokoro(lang=lang, voice=voice)

    # Kokoro returns a generator of (graphemes, phonemes, audio_tensor) tuples
    chunks = []
    for _, _, audio in kpipe(text, voice=voice, speed=speed, split_pattern=None):
        # audio is a torch.Tensor on CPU; convert to numpy
        arr = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
        arr = arr.astype(np.float32)
        chunks.append(arr)

    if not chunks:
        # Return 0.5 s of silence as a fallback
        return np.zeros(TARGET_SR // 2, dtype=np.float32)

    audio_full = np.concatenate(chunks)
    audio_16k  = _resample(audio_full, KOKORO_SR, TARGET_SR)
    return audio_16k


def synthesize_streaming(text: str,
                         voice: str = "af_heart",
                         lang: str = "a",
                         speed: float = 0.63,
                         chunk_size_ms: int = 200):
    """
    Generator that yields 16 kHz mono float32 PCM chunks as they are produced.

    Useful when you want to start feeding FlashHead before the full sentence
    has been synthesised (lower TTFA – time to first audio).

    Yields
    ------
    np.ndarray, shape (chunk_samples,), dtype float32
    """
    kpipe = _get_kokoro(lang=lang, voice=voice)
    chunk_samples = int(TARGET_SR * chunk_size_ms / 1000)

    leftover = np.array([], dtype=np.float32)

    for _, _, audio in kpipe(text, voice=voice, speed=speed, split_pattern=None):
        arr = audio.numpy() if hasattr(audio, "numpy") else np.array(audio)
        arr = _resample(arr.astype(np.float32), KOKORO_SR, TARGET_SR)
        leftover = np.concatenate([leftover, arr])

        while len(leftover) >= chunk_samples:
            yield leftover[:chunk_samples]
            leftover = leftover[chunk_samples:]

    # flush remaining
    if len(leftover) > 0:
        yield leftover


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf
    audio = synthesize("Hello! I am your live avatar. How can I help you today?")
    sf.write("test_tts.wav", audio, TARGET_SR)
    print(f"Saved test_tts.wav  ({len(audio)/TARGET_SR:.2f}s @ {TARGET_SR}Hz)")
