import hashlib
from pathlib import Path

import numpy as np

from ms_model.io.masks import FrameMasks

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "masks"


def _cache_key(npz_path: str, ckpt_path: str) -> Path:
    parts = [str(Path(p).resolve()) for p in (npz_path, ckpt_path)]
    h = hashlib.md5("|".join(parts).encode()).hexdigest()
    return CACHE_DIR / f"{h}.npz"


def load_cached_masks(npz_path: str, ckpt_path: str):
    key = _cache_key(npz_path, ckpt_path)
    if not key.exists():
        return None
    data = np.load(key)
    return FrameMasks(masks=data["masks"], ts=data["ts"])


def save_masks(npz_path: str, ckpt_path: str, fm: FrameMasks) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(npz_path, ckpt_path)
    np.savez_compressed(key, masks=fm.masks, ts=fm.ts)
