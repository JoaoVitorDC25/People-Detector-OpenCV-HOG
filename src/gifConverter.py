import cv2 as cv
import imageio
import os

from config import AVI_PATH, GIF_PATH, FPS_GIF

def convert_avi_to_gif(frame_skip=2, resize=None): 
    """Converte um vídeo AVI para GIF."""
    cap = cv.VideoCapture(AVI_PATH)

    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {AVI_PATH}")
        return

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_skip == 0:
            if resize is not None:
                frame = cv.resize(frame, resize)

            # converte BGR -> RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    cap.release()

    if not frames:
        print("Nenhum frame foi extraído para gerar o GIF.")
        return

    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)

    imageio.mimsave(GIF_PATH, frames, fps=FPS_GIF)
    print(f"GIF salvo em: {GIF_PATH}")