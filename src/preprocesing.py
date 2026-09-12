import cv2 as cv

def preprocess_frame(frame):
    # Gira 90° no sentido horário
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    
    # resizing for faster detection
    frame_p = cv.resize(frame, (480, 640))
    # using a greyscale picture, also for faster detection
    gray = cv.cvtColor(frame_p, cv.COLOR_BGR2GRAY)
    return frame_p, gray