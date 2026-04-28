"""
config.py
---------
Runtime configuration loader. Reads config.yaml at startup; supports
in-process patching for the /config POST endpoint (e.g. live voice change).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIG_FILE = Path(__file__).parent / "config.yaml"
_lock = threading.RLock()

_DEFAULTS: DictConfig = OmegaConf.create({
    "tts":      {"voice": "am_michael", "speed": 1.0, "language": "a"},
    "enhancer": {"enabled": True, "model_path": "experiments/pretrained_models/GFPGANv1.3.pth"},
    "chunking": {"enabled": True, "min_chars": 5},
    "avatar":   {"cache_dir": "./cache/avatars", "bbox_shift": 0, "extra_margin": 10},
    "unet":     {"batch_size": 8, "fps": 25, "use_float16": True, "compile": True,
                 "audio_padding_left": 2, "audio_padding_right": 2},
    "server":   {"host": "0.0.0.0", "port": 7860},
})

_cfg: DictConfig = OmegaConf.create({})


def load() -> DictConfig:
    """Load config.yaml, merge over defaults, return the merged config."""
    global _cfg
    with _lock:
        user = OmegaConf.load(_CONFIG_FILE) if _CONFIG_FILE.exists() else OmegaConf.create({})
        _cfg = OmegaConf.merge(_DEFAULTS, user)
        return _cfg


def get() -> DictConfig:
    with _lock:
        return _cfg


def patch(updates: dict) -> DictConfig:
    """Deep-merge `updates` into the live config. Returns the new config."""
    global _cfg
    with _lock:
        _cfg = OmegaConf.merge(_cfg, OmegaConf.create(updates))
        return _cfg


def to_dict() -> dict[str, Any]:
    with _lock:
        return OmegaConf.to_container(_cfg, resolve=True)  # type: ignore[return-value]


# Auto-load on import
load()
