import tempfile
from pathlib import Path

import gradio as gr

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask
from ms_model.viz.mask import make_overlay_frames

from app.backend.inference import get_or_run_inference
from app.backend.video_utils import make_browser_video

MS_MODEL_ROOT = Path(__file__).parent.parent.parent.parent / "MS_Model"


def _scan_configs():
    configs = sorted(MS_MODEL_ROOT.glob("configs/*.yaml"))
    return [(p.name, str(p)) for p in configs]


def _scan_checkpoints():
    pts = sorted(MS_MODEL_ROOT.glob("checkpoints/**/*.pt"))
    return [(str(p.relative_to(MS_MODEL_ROOT / "checkpoints")), str(p)) for p in pts]


def _predict(npz_path: str, ckpt_path: str, yaml_path: str):
    npz_path = npz_path.strip()
    ckpt_path = (ckpt_path or "").strip()
    yaml_path = (yaml_path or "").strip() or None

    if not npz_path or not ckpt_path:
        return "Erreur : renseigner le .npz et le checkpoint."

    try:
        get_or_run_inference(npz_path, ckpt_path, yaml_path)
        return "Masque prédit et mis en cache."
    except Exception as e:
        return f"Erreur : {e}"


def _visualize(npz_path: str, ckpt_path: str, yaml_path: str, thicken: bool, radius: int, fps: int):
    npz_path = npz_path.strip()
    ckpt_path = (ckpt_path or "").strip()
    yaml_path = (yaml_path or "").strip() or None

    if not npz_path or not ckpt_path:
        return None, None

    ea = load_events(npz_path)
    fm_gt = load_evimo_mask(npz_path)
    fm_pred = get_or_run_inference(npz_path, ckpt_path, yaml_path)

    fm_pred_disp = thicken_mask(fm_pred, int(radius)) if thicken else fm_pred
    fm_gt_disp = thicken_mask(fm_gt, int(radius)) if thicken else fm_gt

    left_frames = make_overlay_frames(ea, fm_pred_disp)
    right_frames = make_overlay_frames(ea, fm_gt_disp)

    left_path = Path(tempfile.mktemp(suffix="_pred.mp4"))
    right_path = Path(tempfile.mktemp(suffix="_gt.mp4"))

    make_browser_video(left_frames, left_path, fps=int(fps))
    make_browser_video(right_frames, right_path, fps=int(fps))

    return str(left_path), str(right_path)


def build():
    configs = _scan_configs()
    checkpoints = _scan_checkpoints()

    with gr.Column():
        gr.Markdown("### Générer le masque")

        npz_input = gr.Textbox(
            value="/home/arthur/IPAL/arthur_ipal/datasets/EV-IMO/eval/box/npz/seq_00.npz",
            label="Chemin .npz",
        )

        ckpt_input = gr.Dropdown(
            choices=checkpoints,
            value=checkpoints[0][1] if checkpoints else None,
            label="Checkpoint .pt",
        )

        yaml_input = gr.Dropdown(
            choices=configs,
            value=None,
            label="Config .yaml (uniquement si le checkpoint n'a pas de config embarquée)",
        )

        predict_btn = gr.Button("Prédire")
        status_out = gr.Textbox(label="Statut", interactive=False)

        predict_btn.click(
            fn=_predict,
            inputs=[npz_input, ckpt_input, yaml_input],
            outputs=[status_out],
        )

        gr.Markdown("---")
        gr.Markdown("### Visualiser")

        with gr.Row():
            thicken_cb = gr.Checkbox(value=False, label="Thicken mask")
            radius_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Rayon")
            fps_input = gr.Number(value=20, label="FPS", minimum=1, maximum=60)

        viz_btn = gr.Button("Visualiser")

        with gr.Row():
            video_pred = gr.Video(label="Masque prédit")
            video_gt = gr.Video(label="Masque GT")

        viz_btn.click(
            fn=_visualize,
            inputs=[npz_input, ckpt_input, yaml_input, thicken_cb, radius_slider, fps_input],
            outputs=[video_pred, video_gt],
        )
