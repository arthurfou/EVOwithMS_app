"""Stub d'appel au pipeline DEVO (estimation de pose).

L'intégration réelle (run_voxel + chargement du réseau DEVO) sera ajoutée
quand les checkpoints DEVO seront disponibles localement.
"""

from pathlib import Path


def run_devo(npz_path: str, devo_ckpt_path: str) -> None:
    """Lance DEVO sur un fichier npz avec le checkpoint donné.

    Lève une exception si les inputs sont invalides.
    Stub pour l'instant — intégration réelle à venir.
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Fichier npz introuvable : {npz_path}")
    if not Path(devo_ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint DEVO introuvable : {devo_ckpt_path}")

    # TODO: appel réel
    # from devo.config import cfg
    # from utils.eval_utils import run_voxel
    # from utils.load_utils import evimo_evs_iterator
    # cfg.merge_from_file("config/eval_evimo.yaml")
    # net = torch.load(devo_ckpt_path)
    # run_voxel(npz_path, cfg, net, iterator=evimo_evs_iterator(...))
