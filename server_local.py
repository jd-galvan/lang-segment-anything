from io import BytesIO
import numpy as np
from PIL import Image
from lang_sam.lang_sam import LangSAM
from lang_sam.utils import draw_image
from opencv.opencv import detect_faces

model = LangSAM(sam_type="sam2.1_hiera_small", device="cuda")

def detect_faces(image_path: str):
    """
    Función que detecta la presencia de rostros para excluirlos del futuro impainting
    
    Parámetros:
      - image_path: Ruta al archivo de imagen de entrada.
      
    Retorna:
      - Una imagen PIL con los resultados dibujados.
    """

    try:
        image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error al abrir la imagen: {e}")

    # Realiza la predicción.
    results = detect_faces(image_path=image_path)

    output_image = Image.fromarray(np.uint8(results)).convert("RGB")
    return output_image


def predict_local(sam_type: str, box_threshold: float, text_threshold: float, image_path: str, text_prompt: str) -> Image.Image:
    """
    Función que realiza la predicción usando el modelo LangSAM de forma local.
    
    Parámetros:
      - sam_type: Tipo de modelo SAM a utilizar.
      - box_threshold: Umbral para detección de cajas.
      - text_threshold: Umbral para detección de texto.
      - image_path: Ruta al archivo de imagen de entrada.
      - text_prompt: Texto de entrada para la predicción.
      
    Retorna:
      - Una imagen PIL con los resultados dibujados.
    """
    # Si se solicita otro modelo, se actualiza el modelo SAM.
    if sam_type != model.sam_type:
        print(f"Updating SAM model type to {sam_type}")
        model.sam.build_model(sam_type)

    try:
        image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error al abrir la imagen: {e}")

    # Realiza la predicción.
    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=[text_prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    results = results[0]

    # Si no se detectaron máscaras, retorna la imagen original.
    if not len(results["masks"]):
        print("No masks detected. Returning original image.")
        return image_pil

    # Dibuja los resultados sobre la imagen.
    image_array = np.asarray(image_pil)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
    return output_image