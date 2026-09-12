from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#/application.py
CAMERAS_INDICES = (2,1,0)
FPS_AVI = 15

WIN_STRIDE = (4, 4)

VIDEO_INPUT_DIR = BASE_DIR / "videos" / "input"
VIDEO_INPUT_PATH = VIDEO_INPUT_DIR / "Walk1.mpg"

#/gifConverter.py
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_OUTPUT_DIR = BASE_DIR / "videos" / "output"
AVI_PATH = VIDEO_OUTPUT_DIR / "people_detection.avi"
GIF_PATH = VIDEO_OUTPUT_DIR / "people_detection.gif"
FPS_GIF = 20