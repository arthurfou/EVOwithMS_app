import gradio as gr

from app.backend.video_cache import get_or_make_video


def _generate(npz_path: str, n_frames: int, fps: int):
    if not npz_path.strip():
        return None
    return get_or_make_video(npz_path.strip(), n_frames=int(n_frames), fps=int(fps))


def build():
    with gr.Column():
        with gr.Row():
            npz_input = gr.Textbox(
                value="/home/arthur/IPAL/arthur_ipal/datasets/EV-IMO/eval/box/npz/seq_00.npz",
                label="Chemin du fichier .npz",
                scale=4,
            )
        with gr.Row():
            fps_input = gr.Number(value=20, label="FPS", minimum=1, maximum=60, scale=1)
            n_frames_input = gr.Number(
                value=100,
                label="N frames (ignoré si frame_ts dispo dans le npz)",
                minimum=10,
                maximum=2000,
                scale=2,
            )

        gen_btn = gr.Button("Générer la vidéo")
        video_out = gr.Video(label="Events (rouge=ON, bleu=OFF)")

        gen_btn.click(
            fn=_generate,
            inputs=[npz_input, n_frames_input, fps_input],
            outputs=[video_out],
        )
