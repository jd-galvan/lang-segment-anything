import os
from PIL import Image
import gradio as gr
import cv2
import numpy as np

from lang_sam import SAM_MODELS
from server_local import generate_mask, detect_faces

def detect_faces_state(image_path):
    """
    Función "wrapper" que llama a detect_faces y retorna la máscara dos veces:
    - Una para mostrar en pantalla.
    - Otra para almacenar en gr.State y usarla en otro botón.
    """
    detect_face_mask = detect_faces(image_path)
    detect_face_image = Image.fromarray(np.uint8(detect_face_mask)).convert("RGB")
    return detect_face_image, detect_face_mask

def process_mask(mask, image_path):
    """
    Función que recibe la máscara de rostros (mask) y el path de la imagen original.
    Coloca los píxeles negros de 'mask' en las mismas ubicaciones de 'generated_mask'.
    
    Retorna:
      - processed_mask: imagen PIL resultante para mostrar.
      - combined_mask: array numpy con la máscara combinada, para uso posterior.
    """
    generated_mask = generate_mask('sam2.1_hiera_small', image_path)
    combined_mask = generated_mask.copy()
    combined_mask[mask == 0] = 0
    processed_mask = Image.fromarray(combined_mask)
    return processed_mask, combined_mask

def final_process(combined_mask):
    """
    Función que toma la combined_mask y realiza un procesamiento final.
    En este ejemplo se invierten los colores de la máscara.
    """
    final = 255 - combined_mask
    return Image.fromarray(final)

with gr.Blocks(title="LangSAM Local") as blocks:
    # Estados para almacenar máscaras intermedias
    mask_state = gr.State()             # Almacena la máscara de detección de rostros.
    combined_mask_state = gr.State()      # Almacena la máscara combinada (resultado de process_mask).
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Detected Faces Mask")
    
    text_prompt = gr.Textbox(lines=1, label="Text Prompt")
    
    # Botones para las dos primeras etapas
    submit_btn = gr.Button("Run Face Detection")
    process_btn = gr.Button("Run Generate Mask")
    
    # Al hacer clic en "Run Face Detection"
    submit_btn.click(
        fn=detect_faces_state,
        inputs=[image_input],
        outputs=[output_image, mask_state],
    )
    
    # Aquí se organiza en una columna la salida "Processed Mask", el botón "Run Final Process" y su resultado.
    with gr.Column():
        processed_image_output = gr.Image(type="pil", label="Processed Mask")
        # Botón para procesar la máscara combinada.
        final_btn = gr.Button("Run Final Process")
        final_image_output = gr.Image(type="pil", label="Final Processed Mask")
    
    # Al hacer clic en "Run Generate Mask", se procesa la máscara detectada y se guarda en combined_mask_state.
    process_btn.click(
        fn=process_mask,
        inputs=[mask_state, image_input],
        outputs=[processed_image_output, combined_mask_state],
    )
    
    # Al hacer clic en "Run Final Process", se toma la máscara combinada y se muestra el resultado final.
    final_btn.click(
        fn=final_process,
        inputs=[combined_mask_state],
        outputs=final_image_output,
    )

blocks.launch(inline=True, share=True)
