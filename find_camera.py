"""Show the available Windows cameras and print usable device indexes."""

import cv2


def main() -> None:
    for index in range(10):
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not camera.isOpened():
                continue

            success, frame = camera.read()
            if not success:
                continue

            print(f"Camera found at index: {index}")
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(1500)
            cv2.destroyWindow(f"Camera {index}")
        finally:
            camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
