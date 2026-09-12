import numpy as np
import cv2 as cv

from config import CAMERAS_INDICES, AVI_PATH, FPS_AVI, VIDEO_INPUT_PATH, WIN_STRIDE
import gifConverter
from preprocesing import preprocess_frame

# initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

def open_camera(indices: tuple[int, ...]) -> tuple[cv.VideoCapture | None, int | None]:
    """Abre a primeira câmera disponível dentre os índices configurados."""
    for index in indices:
        capture = cv.VideoCapture(index)

        if capture.isOpened():
            return capture, index

        capture.release()
        print(f"[AVISO] Câmera no índice {index} não respondeu.")

    return None, None  

def draw_boxes(frame, boxes):
    """Desenha retângulos em torno das pessoas detectadas."""
    for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
            
def process_frame(cap: cv.VideoCapture, out: cv.VideoWriter) -> None:
    """Processa os frames capturados da câmera."""
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break
        
        preprocessed_frame, gray = preprocess_frame(frame)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(gray, winStride=WIN_STRIDE, scale=1.03)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        draw_boxes(preprocessed_frame, boxes)
               
        # Write the output video 
        out.write(preprocessed_frame.astype('uint8'))
        
        # Display the resulting frame
        cv.imshow('frame',preprocessed_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):        
            break

def run():
    """Função principal para executar o programa."""
    # open webcam or USB CAM video stream 
    #cap, index = open_camera(CAMERAS_INDICES)
    
    # unique path to the input video file
    cap = cv.VideoCapture(str(VIDEO_INPUT_PATH))

    
    # the output will be written to output.avi
    out = cv.VideoWriter( str(AVI_PATH), cv.VideoWriter_fourcc(*'MJPG'), FPS_AVI,(480, 640))
 
    try:
        process_frame(cap, out)

    finally:
        # When everything done, release the capture
        cap.release()
        # and release the output
        out.release()
        # finally, close the window
        cv.destroyAllWindows()
        cv.waitKey(1)
        gifConverter.convert_avi_to_gif()
    