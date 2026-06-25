from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.io.masks import FrameMasks
from ms_model.models import build_model
from ms_model.representations.voxel import make_voxel_sequence

from .mask_cache import load_cached_masks, save_masks


def run_inference(npz_path: str, ckpt_path: str, yaml_path: Optional[str] = None) -> FrameMasks:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "config" in ckpt:
        config = ckpt["config"]
    elif yaml_path:
        config = yaml.safe_load(Path(yaml_path).read_text())
    else:
        raise ValueError(
            "Ce checkpoint ne contient pas de config embarquée. "
            "Renseigne le fichier .yaml correspondant."
        )

    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")

    nb_bins = data_cfg.get("nb_time_bins", 5)
    patch_size = data_cfg.get("patch_size", 4)

    ea = load_events(npz_path)
    fm_gt = load_evimo_mask(npz_path)

    voxels = make_voxel_sequence(ea, fm_gt.ts, nb_of_time_bins=nb_bins)  # (T, C, H, W)

    model = build_model(model_name, in_channels=nb_bins, patch_size=patch_size, **model_cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    with torch.no_grad():
        logits = model(voxels.unsqueeze(0))  # (1, T, 1, Hp, Wp)

    probs = torch.sigmoid(logits[0, :, 0])
    binary = (probs > 0.5).numpy().astype(np.uint8)

    binary_full = F.interpolate(
        torch.from_numpy(binary[:, None].astype(np.float32)),
        size=(ea.H, ea.W),
        mode="nearest",
    ).squeeze(1).numpy().astype(np.uint8)

    return FrameMasks(masks=binary_full, ts=fm_gt.ts[:-1])


def get_or_run_inference(npz_path: str, ckpt_path: str, yaml_path: Optional[str] = None) -> FrameMasks:
    fm = load_cached_masks(npz_path, ckpt_path)
    if fm is not None:
        return fm
    fm = run_inference(npz_path, ckpt_path, yaml_path)
    save_masks(npz_path, ckpt_path, fm)
    return fm
