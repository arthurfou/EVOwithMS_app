import subprocess
import tempfile
from pathlib import Path


def reencode_h264(src: Path) -> Path:
    """Re-encode src (mp4v) vers H.264 (libopenh264) lisible par les navigateurs.

    Écrit dans un fichier temporaire et supprime src.
    """
    dst = Path(tempfile.mktemp(suffix=".mp4"))
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    src.unlink(missing_ok=True)
    return dst


def make_browser_video(frames, out_path: Path, fps: int = 20) -> Path:
    """Écrit les frames en mp4v puis re-encode en H.264 vers out_path."""
    from ms_model.viz.video import frames_to_video

    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    frames_to_video(frames, tmp, fps=fps)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink(missing_ok=True)
    return out_path
