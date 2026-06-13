import tempfile
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

H, W = 260, 346

_EVENT_MSG = "uint16 x\nuint16 y\ntime ts\nbool polarity\n"
_EVENT_ARRAY_MSG = (
    "std_msgs/Header header\nuint32 height\nuint32 width\ndvs_msgs/Event[] events\n"
)


def _make_typestore():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    add_types.update(get_types_from_msg(_EVENT_MSG, "dvs_msgs/msg/Event"))
    add_types.update(get_types_from_msg(_EVENT_ARRAY_MSG, "dvs_msgs/msg/EventArray"))
    typestore.register(add_types)
    return typestore


def load_evs(seq_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lit evs.npy et tss_imgs_us.txt depuis un répertoire de séquence filtré.
    Retourne:
      evs       : (N, 4) float64 — [t_µs_relatif, x, y, polarity]
      frame_ts  : (M,) float64  — timestamps µs relatifs de chaque frame
    """
    seq = Path(seq_dir)
    evs = np.load(seq / "evs.npy")
    frame_ts = np.loadtxt(seq / "tss_imgs_us.txt", dtype=np.float64)
    return evs, frame_ts


def load_bag_as_evs(bag_path: str) -> np.ndarray:
    """
    Extrait les events d'un .bag et les retourne en (N, 4) float64
    [t_µs_relatif, x, y, polarity], avec t recalé pour démarrer à 0.
    """
    typestore = _make_typestore()
    rows = []
    with Reader(bag_path) as reader:
        ev_conns = [c for c in reader.connections if c.topic == "/dvs/events"]
        for conn, _, raw in reader.messages(connections=ev_conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            for e in msg.events:
                t_us = e.ts.sec * 1_000_000.0 + e.ts.nanosec / 1_000.0
                rows.append((t_us, e.x, e.y, float(e.polarity)))
    evs = np.array(rows, dtype=np.float64)
    evs[:, 0] -= evs[0, 0]  # recalage relatif à 0
    return evs


def _render_canvas(evs: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    t = evs[:, 0]
    xs = evs[:, 1].astype(np.int32)
    ys = evs[:, 2].astype(np.int32)
    pols = evs[:, 3].astype(bool)
    mask = (t >= t_lo) & (t < t_hi)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[ys[mask & pols], xs[mask & pols]] = [255, 0, 0]
    canvas[ys[mask & ~pols], xs[mask & ~pols]] = [0, 0, 255]
    return canvas


def make_comparison_frames(
    evs_orig: np.ndarray,
    evs_filt: np.ndarray,
    frame_ts: np.ndarray,
) -> list[np.ndarray]:
    """
    Génère des frames côte à côte (original gauche | filtré droite).
    frame_ts sert de timeline commune (issu de tss_imgs_us.txt).
    """
    boundaries = np.concatenate([[0.0], frame_ts])
    frames = []
    for i in range(1, len(boundaries)):
        t_lo, t_hi = boundaries[i - 1], boundaries[i]
        left = _render_canvas(evs_orig, t_lo, t_hi)
        right = _render_canvas(evs_filt, t_lo, t_hi)
        frame = np.concatenate([left, right], axis=1)
        frames.append(frame)
    return frames


def make_video(frames: list[np.ndarray], fps: int = 20) -> str:
    """Génère un MP4 et retourne son chemin temporaire."""
    h, w = frames[0].shape[:2]
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path
