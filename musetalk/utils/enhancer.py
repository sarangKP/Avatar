# musetalk/utils/enhancer.py
import cv2
import numpy as np

_gfpgan_enhancer = None
_enabled = True
_warned_unavailable = False


def set_enabled(enabled: bool) -> None:
    """Toggle GFPGAN enhancement at runtime (called from /config endpoint)."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def _get_gfpgan():
    global _gfpgan_enhancer
    if _gfpgan_enhancer is None:
        from gfpgan import GFPGANer
        _gfpgan_enhancer = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
        print("[Enhancer] GFPGAN loaded")
    return _gfpgan_enhancer


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Enhance face region using GFPGAN. No-op if disabled or unavailable."""
    global _warned_unavailable
    if not _enabled:
        return frame
    try:
        enhancer = _get_gfpgan()
        _, _, enhanced = enhancer.enhance(
            frame,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        return enhanced if enhanced is not None else frame
    except ImportError:
        if not _warned_unavailable:
            print("[Enhancer] gfpgan not installed — skipping enhancement. "
                  "Install with: uv add gfpgan")
            _warned_unavailable = True
        return frame
    except Exception as e:
        if not _warned_unavailable:
            print(f"[Enhancer] GFPGAN failed, returning original: {e}")
            _warned_unavailable = True
        return frame
