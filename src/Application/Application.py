from src.Util.Detector import Detector
import cv2


def main():
    capture = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        success, img = capture.read()
        detector.annotate_img(img)
        img = cv2.flip(img, 1)
        cv2.imshow("Test", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
