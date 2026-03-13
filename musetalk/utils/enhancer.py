# musetalk/utils/enhancer.py
import cv2
import numpy as np

_gfpgan_enhancer = None

def _get_gfpgan():
    global _gfpgan_enhancer
    if _gfpgan_enhancer is None:
        from gfpgan import GFPGANer
        _gfpgan_enhancer = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        print("[Enhancer] GFPGAN loaded")
    return _gfpgan_enhancer

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Enhance face region using GFPGAN."""
    try:
        enhancer = _get_gfpgan()
        _, _, enhanced = enhancer.enhance(
            frame,
            has_aligned=False,
            only_center_face=True,
            paste_back=True
        )
        return enhanced if enhanced is not None else frame
    except Exception as e:
        print(f"[Enhancer] GFPGAN failed, returning original: {e}")
        return frame