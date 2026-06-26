import tempfile
from pathlib import Path

import gradio as gr

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask
from ms_model.viz.mask import make_overlay_frames

from app.backend.inference import get_or_run_inference
from app.backend.video_utils import make_browser_video
from app.tabs.segmentation import _scan_configs, _scan_checkpoints

DATASETS_ROOT = Path(__file__).parent.parent.parent.parent / "datasets"
_FPS = 20


def _scan_npz():
    return sorted(DATASETS_ROOT.rglob("*.npz"))


def _predict_and_visualize(npz_path: str, ckpt_path: str, yaml_path: str, thicken: bool, radius: int):
    ckpt_path = (ckpt_path or "").strip()
    yaml_path = (yaml_path or "").strip() or None

    if not ckpt_path:
        return "Erreur : sélectionner un checkpoint.", None, None

    try:
        ea = load_events(npz_path)
        fm_gt = load_evimo_mask(npz_path)
        fm_pred = get_or_run_inference(npz_path, ckpt_path, yaml_path)

        fm_pred_disp = thicken_mask(fm_pred, int(radius)) if thicken else fm_pred
        fm_gt_disp = thicken_mask(fm_gt, int(radius)) if thicken else fm_gt

        left_frames = make_overlay_frames(ea, fm_pred_disp)
        right_frames = make_overlay_frames(ea, fm_gt_disp)

        left_path = Path(tempfile.mktemp(suffix="_pred.mp4"))
        right_path = Path(tempfile.mktemp(suffix="_gt.mp4"))

        make_browser_video(left_frames, left_path, fps=_FPS)
        make_browser_video(right_frames, right_path, fps=_FPS)

        return "OK", str(left_path), str(right_path)
    except Exception as e:
        return f"Erreur : {e}", None, None


def build():
    npz_files = _scan_npz()
    all_paths = [str(p) for p in npz_files]
    choices = [(str(p.relative_to(DATASETS_ROOT)), str(p)) for p in npz_files]

    configs = _scan_configs()
    checkpoints = _scan_checkpoints()

    with gr.Column():
        gr.Markdown("### Checkpoint")

        ckpt_input = gr.Dropdown(
            choices=checkpoints,
            value=checkpoints[0][1] if checkpoints else None,
            label="Checkpoint .pt",
        )
        yaml_input = gr.Dropdown(
            choices=configs,
            value=None,
            label="Config .yaml (si le checkpoint n'a pas de config embarquée)",
        )

        gr.Markdown("---")
        gr.Markdown("### Sélection des fichiers")

        file_selector = gr.CheckboxGroup(
            choices=choices,
            value=[],
            label="Fichiers .npz disponibles",
        )

        gr.Markdown("---")
        gr.Markdown("### Prédiction par fichier")

        file_groups = []
        btn_list, status_list, thicken_list, radius_list = [], [], [], []
        video_pred_list, video_gt_list = [], []

        for npz_path_str, rel_label in [(str(p), str(p.relative_to(DATASETS_ROOT))) for p in npz_files]:
            grp = gr.Group(visible=False)
            with grp:
                gr.Markdown(f"#### `{rel_label}`")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        btn = gr.Button("Predict & Visualize", variant="primary")
                    with gr.Column(scale=2):
                        status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=1):
                        thicken_cb = gr.Checkbox(value=False, label="Thicken Mask")
                    with gr.Column(scale=1):
                        radius_sl = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Radius")
                with gr.Row():
                    video_pred = gr.Video(label="Masque prédit")
                    video_gt = gr.Video(label="Masque GT")

            file_groups.append(grp)
            btn_list.append(btn)
            status_list.append(status)
            thicken_list.append(thicken_cb)
            radius_list.append(radius_sl)
            video_pred_list.append(video_pred)
            video_gt_list.append(video_gt)

        def _update_visibility(selected):
            return [gr.update(visible=(p in selected)) for p in all_paths]

        file_selector.change(
            fn=_update_visibility,
            inputs=[file_selector],
            outputs=file_groups,
        )

        for i, npz_path_str in enumerate(all_paths):
            npz_state = gr.State(npz_path_str)
            btn_list[i].click(
                fn=_predict_and_visualize,
                inputs=[npz_state, ckpt_input, yaml_input, thicken_list[i], radius_list[i]],
                outputs=[status_list[i], video_pred_list[i], video_gt_list[i]],
            )
