import os
from PIL import Image
import gradio as gr
import cv2
import numpy as np

from lang_sam import SAM_MODELS
from server_local import generate_mask, detect_faces, generate_impaint

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
      - processed_mask: imagen PIL resultante para mostrar en "Processed Mask".
    """
    generated_mask = generate_mask('sam2.1_hiera_small', image_path)
    # Copiamos la máscara generada para no modificar el original.
    combined_mask = generated_mask.copy()
    # En las posiciones donde 'mask' es negro (0), ponemos 0 en combined_mask.
    combined_mask[mask == 0] = 0
    processed_mask = Image.fromarray(combined_mask)
    return processed_mask

def final_process(original_image_path, processed_mask):
    """
    Función que recibe la ruta de la imagen original y la imagen de "Processed Mask".
    """

    new_image = generate_impaint(original_image_path, processed_mask)
    print("RESULTADO")
    print(type(new_image))
    print(new_image)

    return Image.fromarray(new_image)

with gr.Blocks(title="LangSAM Local") as blocks:
    # Estado para almacenar la máscara de detección (array numpy)
    mask_state = gr.State()
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Detected Faces Mask")
    
    text_prompt = gr.Textbox(lines=1, label="Text Prompt")
    
    # Botones para las dos primeras etapas
    submit_btn = gr.Button("Run Face Detection")
    process_btn = gr.Button("Run Generate Mask")
    
    # Se organiza en una columna la salida "Processed Mask", el botón "Run Final Process" y su resultado.
    with gr.Column():
        processed_image_output = gr.Image(type="pil", label="Processed Mask")
        final_btn = gr.Button("Run Generate Impainting")
        final_image_output = gr.Image(type="pil", label="Final Processed Mask")
    
    # Al hacer clic en "Run Face Detection", se muestra la máscara detectada y se guarda en mask_state.
    submit_btn.click(
        fn=detect_faces_state,
        inputs=[image_input],
        outputs=[output_image, mask_state],
    )
    
    # Al hacer clic en "Run Generate Mask", se toma mask_state y la ruta de la imagen para generar la máscara combinada.
    process_btn.click(
        fn=process_mask,
        inputs=[mask_state, image_input],
        outputs=processed_image_output,
    )
    
    # El botón "Run Final Process" está ubicado debajo de "Processed Mask" y toma
    # la ruta de la imagen original y la imagen de "Processed Mask" para generar el resultado final.
    final_btn.click(
        fn=final_process,
        inputs=[image_input, processed_image_output],
        outputs=final_image_output,
    )

blocks.launch(inline=True, share=True)
