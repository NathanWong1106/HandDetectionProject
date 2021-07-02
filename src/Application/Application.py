from src.Util.Detector import Detector
import cv2
import mediapipe as mp
from src.Training.Model import GestureModel


def main():
    capture = cv2.VideoCapture(0)
    detector = Detector()
    predictor = GestureModel()

    while True:
        success, img = capture.read()
        img = cv2.flip(img, 1)

        hands = detector.get_landmarks(img).multi_hand_landmarks

        if hands is not None:
            for hand in hands:
                gesture = predictor.get_gesture_prediction(img, hand)
                detector.annotate_hand(img, hand, gesture)

        cv2.imshow("Test", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
