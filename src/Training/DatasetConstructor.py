import array
import math
from typing import TextIO

import mediapipe as mp
from src.Util.Detector import Detector
import cv2
import csv
import pandas as pd


def main():
    set_name = input("Dataset Name: ")
    append = "a" if input("Append to file (yes/no): ") == "yes" else "w"

    capture = cv2.VideoCapture(0)
    detector = Detector()

    capture_landmarks = False

    with open(f'./Data/{set_name}.csv', append, newline="") as file:

        header_arr = get_std_header()
        header_writer = csv.writer(file)
        header_writer.writerow(header_arr)

        while True:
            success, img = capture.read()
            key = cv2.waitKey(1)

            # # When 'c' is pressed then add the points to the dataset
            # if key == ord('c'):
            #     landmarks = detector.get_landmarks(img).multi_hand_landmarks
            #     write_landmarks_to_dataset(img, landmarks, file, set_name)
            #
            # elif key == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

            if key == ord('c'):
                capture_landmarks = not capture_landmarks

            landmarks = detector.get_landmarks(img).multi_hand_landmarks

            if landmarks is not None and capture_landmarks:
                landmark_arr = [landmarks]
                for flipIndex in range(-1, 2):
                    landmark_arr.append(detector.get_landmarks(cv2.flip(img, flipIndex)).multi_hand_landmarks)

                for elem in landmark_arr:
                    write_landmarks_to_dataset(img, elem, file, set_name)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            detector.annotate_img(img)
            img = cv2.flip(img, 1)
            cv2.imshow("Test", img)


def get_std_header():
    header_arr = []

    for index in mp.solutions.hands.HandLandmark:
        header_arr.append(f"x{index}")
        header_arr.append(f"y{index}")
    header_arr.append("gesture")

    return header_arr


def write_landmarks_to_dataset(img, landmarks, file: TextIO, gesture_name: str):
    """
    Writes the normalized and relative landmark location from the wrist to use as training data
    :param img: numpy array representing a BGR image from cv2
    :param landmarks: landmarks returned from mediapipe
    :param file: file to write to
    :param gesture_name: gesture name
    :return: None
    """
    writer = csv.writer(file)

    if landmarks is not None:
        for hand in landmarks:
            arr = get_normalized_relative_landmarks(img, hand)
            arr.append(gesture_name)
            writer.writerow(arr)
        print("Data Written")


def get_normalized_relative_landmarks(img, hand) -> array.ArrayType:
    height, width, channels = img.shape
    max_x, max_y = -math.inf, -math.inf

    arr = []
    wrist_landmark = hand.landmark[mp.solutions.hands.HandLandmark.WRIST]
    wrist_coords = (wrist_landmark.x * width, wrist_landmark.y * height)

    for index in mp.solutions.hands.HandLandmark:
        # Get the relative coordinates of each landmark from the wrist
        wrist_relative_coords = get_relative_coord(wrist_coords,
                                                   (hand.landmark[index].x * width,
                                                    hand.landmark[index].y * height))

        max_x = abs(wrist_relative_coords[0]) if abs(wrist_relative_coords[0]) > max_x else max_x
        max_y = abs(wrist_relative_coords[1]) if abs(wrist_relative_coords[1]) > max_y else max_y

        arr.append(wrist_relative_coords[0])
        arr.append(wrist_relative_coords[1])

    # Normalize values in the array by maximum absolute value of x,y respectively
    for index, val in enumerate(arr):
        if index % 2 == 0:
            arr[index] = arr[index] / max_x
        else:
            arr[index] = arr[index] / max_y

    return arr


def get_relative_coord(origin: tuple, point: tuple) -> tuple:
    """
    Returns the coordinate of the second point as if the first point was the origin
    """
    return point[0] - origin[0], point[1] - origin[1]


if __name__ == "__main__":
    main()
