import os
from PIL import Image
import gradio as gr

from lang_sam import SAM_MODELS

from lang_sam.server import predict_local, detect_faces

with gr.Blocks(title="LangSAM Local") as blocks:
    with gr.Row():
        sam_model_choices = gr.Dropdown(
            choices=["sam2.1_hiera_small"], 
            label="SAM Model", 
            value="sam2.1_hiera_small"
        )
        box_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Box Threshold")
        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text Threshold")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Output Image")
    text_prompt = gr.Textbox(lines=1, label="Text Prompt")
    submit_btn = gr.Button("Run Prediction")

    submit_btn.click(
        fn=detect_faces,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt],
        outputs=output_image,
    )

    examples = [
        [
            "sam2.1_hiera_small",
            0.32,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi. watermelon. blueberry.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food.",
        ],
    ]
    gr.Examples(
        examples=examples,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt],
        outputs=output_image,
    )

if __name__ == "__main__":
    # Lanza la interfaz de Gradio de forma inline, por ejemplo, en Google Colab.
    blocks.launch(inline=True, share=True)