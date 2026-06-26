import gradio as gr
from app.tabs import event_viewer, segmentation, segmentation_dataset, devo_pipeline

with gr.Blocks(title="Motion Segmentation for Event Visual Odometry") as demo:
    gr.Markdown("# Motion Segmentation for Event Visual Odometry")

    with gr.Tabs():
        with gr.Tab("Event Viewer"):
            event_viewer.build()

        with gr.Tab("Motion Segmentation"):
            segmentation.build()

        with gr.Tab("Motion Segmentation dataset"):
            segmentation_dataset.build()

        with gr.Tab("DEVO Pipeline"):
            devo_pipeline.build()

if __name__ == "__main__":
    demo.launch(share=False)
