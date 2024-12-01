from configs.logger import logger
from recognition import load_known_faces, recognize_faces
from camera import start_camera, get_frame

YOUR_NAME = ""


def main():
    logger.info("FaceForge starts")

    known_images = [
        (YOUR_NAME, "./inputs/<file-name>.jpg"),
    ]

    known_faces = load_known_faces(known_images)
    logger.debug(f"known_faces: {known_faces}")

    video_capture = start_camera()
    logger.debug(f"video_capture: {video_capture}")
    logger.info("video is on: press CTRL+C to exit")

    try:
        while True:
            ret, frame = get_frame(video_capture)
            logger.debug(f"ret: {ret}")

            if not ret:
                break

            results = recognize_faces(frame, known_faces)
            logger.debug(f"results recognize_faces: {results}")

    finally:
        logger.debug("Exit webcam")
        logger.info("Release resources")


if __name__ == "__main__":
    main()
