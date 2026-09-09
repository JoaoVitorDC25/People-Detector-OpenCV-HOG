import numpy as np
import cv2 as cv

from config import CAMERAS_INDICES

#def process():
    
    
#def process_frames():
  
def open_camera(indices: tuple[int, ...]) -> tuple[cv.VideoCapture | None, int | None]:
    """Abre a primeira câmera disponível dentre os índices configurados."""
    for index in indices:
        capture = cv.VideoCapture(index)

        if capture.isOpened():
            return capture, index

        capture.release()
        print(f"[AVISO] Câmera no índice {index} não respondeu.")

    return None, None  
    
#def run():
    