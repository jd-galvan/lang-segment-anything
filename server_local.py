from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from lang_sam.lang_sam import LangSAM
from stable_diffusion.sd import SD
from opencv.opencv import detect_faces as d_faces

model = LangSAM(sam_type="sam2.1_hiera_small", device="cuda")
sd = SD()

def detect_faces(image_path: str):
    """
    Función que detecta la presencia de rostros para excluirlos del futuro impainting
    
    Parámetros:
      - image_path: Ruta al archivo de imagen de entrada.
      
    Retorna:
      - Una imagen PIL con los resultados dibujados.
    """

    # Realiza la predicción.
    results = d_faces(image_path)

    return results


def generate_mask(sam_type: str, image_path: str):
    """
    Función que realiza la predicción usando el modelo LangSAM de forma local.
    
    Parámetros:
      - sam_type: Tipo de modelo SAM a utilizar.
      - image_path: Ruta al archivo de imagen de entrada.
      
    Retorna:
      - Una máscara dilatada en formato de imagen (array de uint8).
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
        texts_prompt=["photo damage"]  # Prompt para que detecte los daños de las fotos
    )
    results = results[0]

    # Verifica que se haya detectado alguna máscara
    if "masks" not in results or len(results["masks"]) == 0:
        raise ValueError("No se detectaron máscaras en la imagen.")

    # Extrae la primera máscara detectada
    pred_mask = np.array(results["masks"][0])
    
    # Asegúrate de que la máscara sea 2D (alto, ancho)
    if pred_mask.ndim != 2:
        raise ValueError("La máscara detectada no tiene el formato esperado (2D).")
    
    h, w = pred_mask.shape
    # Define el color con canal alfa (RGBA), normalizado entre 0 y 1
    color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Crea una imagen de la máscara con el color definido
    mask_image = pred_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Crear una imagen binaria basada en la máscara usando el canal alfa
    mask_alpha = mask_image[:, :, 3]  # Canal alfa
    binary_mask = np.where(mask_alpha > 0, 255, 0).astype('uint8')

    # Define un kernel para la dilatación
    kernel_size = 30  # Ajusta este tamaño para mayor o menor grosor
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Aplica la dilatación
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    return dilated_mask

def generate_impaint(image_path, mask_path):
    new_image = sd.predict(image_path, mask_path)
    return new_image