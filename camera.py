import cv2


def start_camera():
    """Init webcam."""
    return cv2.VideoCapture(0)


def get_frame(video_capture):
    """Get webcam frame."""
    ret, frame = video_capture.read()
    return ret, frame
