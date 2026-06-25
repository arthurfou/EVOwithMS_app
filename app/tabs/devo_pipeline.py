from pathlib import Path

import gradio as gr

from app.backend.inference import get_or_run_inference
from app.backend.devo_runner import run_devo
from app.tabs.segmentation import _scan_configs, _scan_checkpoints


def _run_pipeline(npz_path: str, ms_ckpt: str, yaml_path: str, devo_ckpt: str):
    npz_path = npz_path.strip()
    ms_ckpt = (ms_ckpt or "").strip()
    yaml_path = (yaml_path or "").strip() or None
    devo_ckpt = devo_ckpt.strip()

    if not npz_path or not ms_ckpt or not devo_ckpt:
        return "Erreur : renseigner le .npz, le checkpoint MS et le checkpoint DEVO."

    try:
        yield "Étape 1/2 — Motion segmentation en cours..."
        get_or_run_inference(npz_path, ms_ckpt, yaml_path)

        yield "Étape 2/2 — Estimation de pose DEVO en cours..."
        run_devo(npz_path, devo_ckpt)

        yield "Pipeline terminé avec succès."
    except Exception as e:
        yield f"Erreur : {e}"


def build():
    configs = _scan_configs()
    checkpoints = _scan_checkpoints()

    with gr.Column():
        gr.Markdown("### Entrées")

        npz_input = gr.Textbox(
            value="/home/arthur/IPAL/arthur_ipal/datasets/EV-IMO/eval/box/npz/seq_00.npz",
            label="Chemin .npz",
        )

        ms_ckpt_input = gr.Dropdown(
            choices=checkpoints,
            value=checkpoints[0][1] if checkpoints else None,
            label="Checkpoint MS (.pt)",
        )

        yaml_input = gr.Dropdown(
            choices=configs,
            value=None,
            label="Config MS .yaml (uniquement si le checkpoint n'a pas de config embarquée)",
        )

        devo_ckpt_input = gr.Textbox(
            value="/home/arthur/IPAL/arthur_ipal/DEVO/checkpoints/devo.pt",
            label="Checkpoint DEVO (.pt)",
        )

        gr.Markdown("---")

        run_btn = gr.Button("Lancer le pipeline", variant="primary")
        status_out = gr.Textbox(label="Statut", interactive=False, lines=3)

        run_btn.click(
            fn=_run_pipeline,
            inputs=[npz_input, ms_ckpt_input, yaml_input, devo_ckpt_input],
            outputs=[status_out],
        )
