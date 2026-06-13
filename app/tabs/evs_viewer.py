import gradio as gr
from app.evs_reader import load_bag_as_evs, load_evs, make_comparison_frames, make_video


def _render(orig_bag: str, filt_dir: str, fps: int):
    evs_orig = load_bag_as_evs(orig_bag)
    evs_filt, frame_ts = load_evs(filt_dir)
    frames = make_comparison_frames(evs_orig, evs_filt, frame_ts)
    video_path = make_video(frames, fps=int(fps))
    info = (
        f"{len(frames)} frames — "
        f"original: {len(evs_orig):,} events — "
        f"filtré: {len(evs_filt):,} events "
        f"({100*len(evs_filt)/len(evs_orig):.1f} % conservés)"
    )
    return video_path, info


def build():
    with gr.Column():
        with gr.Row():
            bag_input = gr.Textbox(
                value="datasets/EV-IMO/eval/tabletop/raw/seq_00/seq_00.bag",
                label="Bag original (.bag)",
                scale=4,
            )
            fps_input = gr.Number(value=20, label="FPS", minimum=1, maximum=60, scale=1)
        filt_input = gr.Textbox(
            value="datasets/EV-IMO_filtered/eval/tabletop/raw/seq_00",
            label="Répertoire séquence filtrée (contient evs.npy + tss_imgs_us.txt)",
        )

        render_btn = gr.Button("Générer la vidéo de comparaison")

        info_out = gr.Textbox(label="Infos", interactive=False)
        video_out = gr.Video(label="Original (gauche)  |  Filtré (droite)")

        render_btn.click(
            fn=_render,
            inputs=[bag_input, filt_input, fps_input],
            outputs=[video_out, info_out],
        )
