import tempfile

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

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


def load_bag(bag_path: str) -> tuple[list, list]:
    """
    Retourne:
      frames       : liste de np.array (H, W) uint8   — frames grises APS
      event_frames : liste de np.array (H, W, 3) uint8 — events alignés par timestamp
    """
    typestore = _make_typestore()
    H, W = 260, 346

    aps_ts, frames = [], []
    ev_ts, ev_canvases = [], []

    with Reader(bag_path) as reader:
        img_conns = [c for c in reader.connections if c.topic == "/dvs/image_raw"]
        for conn, ts, raw in reader.messages(connections=img_conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            aps_ts.append(ts)
            frames.append(img)

        ev_conns = [c for c in reader.connections if c.topic == "/dvs/events"]
        for conn, ts, raw in reader.messages(connections=ev_conns):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            xs = np.array([e.x for e in msg.events], dtype=np.int32)
            ys = np.array([e.y for e in msg.events], dtype=np.int32)
            pols = np.array([e.polarity for e in msg.events], dtype=bool)
            canvas[ys[pols], xs[pols]] = [255, 0, 0]
            canvas[ys[~pols], xs[~pols]] = [0, 0, 255]
            ev_ts.append(ts)
            ev_canvases.append(canvas)

    # Pour chaque frame APS, on prend le paquet d'events le plus proche en temps
    ev_ts_arr = np.array(ev_ts)
    event_frames = [
        ev_canvases[np.argmin(np.abs(ev_ts_arr - t))] for t in aps_ts
    ]

    return frames, event_frames


def make_video(frames: list, event_frames: list, fps: int = 20) -> str:
    """
    Génère un MP4 côte à côte (APS | events) et retourne son chemin temporaire.
    """
    n = len(frames)
    H, W = 260, 346
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (W * 2, H))

    for i in range(n):
        aps_bgr = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        ev_bgr = cv2.cvtColor(event_frames[i], cv2.COLOR_RGB2BGR)
        frame = np.concatenate([aps_bgr, ev_bgr], axis=1)
        writer.write(frame)

    writer.release()
    return out_path
