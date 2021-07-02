import math
from typing import NamedTuple
import mediapipe as mp
import cv2
import numpy


class Detector:

    def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands.Hands(static_image_mode, max_num_hands, min_detection_confidence,
                                                min_tracking_confidence)
        self.drawer = mp.solutions.drawing_utils

    def get_landmarks(self, img: numpy.ndarray) -> NamedTuple:
        res = self.mpHands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return res

    def annotate_hand(self, img: numpy.ndarray, hand, gesture: str):
        height, width, channels = img.shape
        min_x, min_y, max_x, max_y = math.inf, math.inf, 0, 0

        self.drawer.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)

        for point in hand.landmark:
            cur_x, cur_y = point.x * width, point.y * height
            min_x = int(cur_x if cur_x < min_x else min_x)
            max_x = int(cur_x if cur_x > max_x else max_x)
            min_y = int(cur_y if cur_y < min_y else min_y)
            max_y = int(cur_y if cur_y > max_y else max_y)

        x = int((min_x + max_x) / 2)
        y = max_y + 1
        cv2.putText(img, gesture, (x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=2)