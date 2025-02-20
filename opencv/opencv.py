import cv2
import numpy as np


def detect_faces(path_image):
    imagen = cv2.imread(path_image)
    # Convertir la imagen a escala de grises, ya que el detector Haar Cascade trabaja en grises.
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Cargar el clasificador Haar Cascade para detección de rostros.
    # Asegúrate de tener instalado OpenCV y de que el path sea correcto.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detectar rostros en la imagen.
    # Los parámetros scaleFactor y minNeighbors pueden ajustarse según el contexto.
    rostros = face_cascade.detectMultiScale(imagen_gray, scaleFactor=1.1, minNeighbors=5)
    
    # Crear una máscara binaria con el mismo tamaño que la imagen en escala de grises.
    mascara_rostros = np.zeros(imagen_gray.shape, dtype=np.uint8)
    
    # Rellenar la máscara con valor 255 en las regiones donde se detectaron rostros.
    for (x, y, w, h) in rostros:
        cv2.rectangle(mascara_rostros, (x, y), (x+w, y+h), 255, thickness=-1)
    
    return mascara_rostros

