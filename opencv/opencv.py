import cv2
import numpy as np

def detect_faces(path_image):
    imagen = cv2.imread(path_image)
    # Convertir la imagen a escala de grises para la detección.
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Cargar el clasificador Haar Cascade para detección de rostros.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detectar rostros en la imagen.
    rostros = face_cascade.detectMultiScale(imagen_gray, scaleFactor=1.1, minNeighbors=5)
    
    # Crear una máscara binaria en blanco (255).
    mascara_rostros = np.full(imagen_gray.shape, 255, dtype=np.uint8)
  
    # Rellenar con negro (0) las regiones donde se detectaron rostros.
    for (x, y, w, h) in rostros:
        cv2.rectangle(mascara_rostros, (x, y), (x+w, y+h), 0, thickness=-1)
    
    return mascara_rostros
