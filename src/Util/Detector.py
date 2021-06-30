import math
from typing import NamedTuple
import mediapipe as mp
import cv2
import numpy


class Detector:

    def __init__(self, static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands.Hands(static_image_mode, max_num_hands, min_detection_confidence,
                                                min_tracking_confidence)
        self.drawer = mp.solutions.drawing_utils

    def get_landmarks(self, img: numpy.ndarray) -> NamedTuple:
        res = self.mpHands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return res

    def annotate_img(self, img: numpy.ndarray):
        landmarks = self.get_landmarks(img).multi_hand_landmarks

        height, width, channels = img.shape
        min_x, min_y, max_x, max_y = math.inf, math.inf, 0, 0

        if landmarks is not None:
            for hand_landmark in landmarks:
                self.drawer.draw_landmarks(img, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS)

                for point in hand_landmark.landmark:
                    cur_x, cur_y = point.x * width, point.y * height

                    min_x = int(cur_x if cur_x < min_x else min_x)
                    max_x = int(cur_x if cur_x > max_x else max_x)
                    min_y = int(cur_y if cur_y < min_y else min_y)
                    max_y = int(cur_y if cur_y > max_y else max_y)

            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
