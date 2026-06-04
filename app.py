import gradio as gr

with gr.Blocks(title="EVOwithMS") as demo:
    gr.Markdown("# EVOwithMS — Event Visual Odometry with Motion Segmentation")

    with gr.Tabs():
        with gr.Tab("Filtrage EVIMO"):
            gr.Markdown("*À venir*")

        with gr.Tab("Motion Segmentation"):
            gr.Markdown("*À venir*")

        with gr.Tab("DEVO Odometry"):
            gr.Markdown("*À venir*")

if __name__ == "__main__":
    demo.launch()
