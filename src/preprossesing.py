import cv2 as cv

def preprocess_frame(frame):
    
    # resizing for faster detection
    frame = cv.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    return frame, gray