import gradio as gr
from app.tabs import evimo_viewer, evs_viewer

with gr.Blocks(title="EVOwithMS") as demo:
    gr.Markdown("# EVOwithMS — Event Visual Odometry with Motion Segmentation")

    with gr.Tabs():
        with gr.Tab("Filtrage EVIMO"):
            evimo_viewer.build()

        with gr.Tab("EV-IMO Filtered — Events"):
            evs_viewer.build()

        with gr.Tab("Motion Segmentation"):
            gr.Markdown("*À venir*")

        with gr.Tab("DEVO Odometry"):
            gr.Markdown("*À venir*")

if __name__ == "__main__":
    demo.launch(share=True)
