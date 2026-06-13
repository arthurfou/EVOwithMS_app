import gradio as gr
from app.bag_reader import load_bag, make_video


def _load_and_render(bag_path, fps):
    frames, event_frames = load_bag(bag_path)
    video_path = make_video(frames, event_frames, fps=int(fps))
    return video_path


def build():
    with gr.Column():
        with gr.Row():
            bag_input = gr.Textbox(
                value="datasets/EV-IMO/eval/box/raw/seq_00/seq_00.bag",
                label="Chemin du fichier .bag",
                scale=4,
            )
            fps_input = gr.Number(value=20, label="FPS", minimum=1, maximum=60, scale=1)

        load_btn = gr.Button("Générer la vidéo")

        video_out = gr.Video(label="APS  |  Events (rouge=ON, bleu=OFF)")

        load_btn.click(
            fn=_load_and_render,
            inputs=[bag_input, fps_input],
            outputs=[video_out],
        )
