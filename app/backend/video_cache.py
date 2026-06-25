import hashlib
from pathlib import Path

from ms_model.io.loaders import load_events
from ms_model.viz.render import make_frames

from .video_utils import make_browser_video

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "videos"


def _cache_key(npz_path: str, n_frames: int, fps: int) -> Path:
    h = hashlib.md5(str(Path(npz_path).resolve()).encode()).hexdigest()
    return CACHE_DIR / f"{h}_{n_frames}_{fps}.mp4"


def get_or_make_video(npz_path: str, n_frames: int = 100, fps: int = 20) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(npz_path, n_frames, fps)

    if cache_path.exists():
        return str(cache_path)

    ea = load_events(npz_path)
    frames = make_frames(ea, n_frames=n_frames if ea.frame_ts is None else None)
    make_browser_video(frames, cache_path, fps=fps)
    return str(cache_path)
