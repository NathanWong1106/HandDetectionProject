import math
from typing import TextIO

import mediapipe
from src.Util.Detector import Detector
import cv2
import csv


def main():

    set_name = input("Dataset Name: ")
    append = "a" if input("Append to file (yes/no): ") == "yes" else "w"

    capture = cv2.VideoCapture(0)
    detector = Detector()

    with open(f'{set_name}.csv', append) as file:
        while True:
            success, img = capture.read()
            key = cv2.waitKey(1)

            # When 'c' is pressed then add the points to the dataset
            if key == ord('c'):
                landmarks = detector.get_landmarks(img).multi_hand_landmarks
                write_landmarks_to_dataset(img, landmarks, file)

            elif key == ord('q'):
                cv2.destroyAllWindows()
                break

            detector.annotate_img(img)
            img = cv2.flip(img, 1)
            cv2.imshow("Test", img)


def write_landmarks_to_dataset(img, landmarks, file: TextIO):
    """
    Writes the normalized and relative landmark location from the wrist to use as training data
    :param img: numpy array representing a BGR image from cv2
    :param landmarks: landmarks returned from mediapipe
    :param file: file to write to
    :return: None
    """
    writer = csv.writer(file)
    height, width, channels = img.shape
    max_x, max_y = -math.inf, -math.inf

    if landmarks is not None:
        write_arr = []
        for hand_landmark in landmarks:
            arr = []
            wrist_landmark = hand_landmark.landmark[mediapipe.solutions.hands.HandLandmark.WRIST]
            wrist_coords = (wrist_landmark.x * width, wrist_landmark.y * height)

            for index in mediapipe.solutions.hands.HandLandmark:
                # Get the relative coordinates of each landmark from the wrist
                wrist_relative_coords = get_relative_coord(wrist_coords,
                                                           (hand_landmark.landmark[index].x * width,
                                                            hand_landmark.landmark[index].y * height))

                max_x = abs(wrist_relative_coords[0]) if abs(wrist_relative_coords[0]) > max_x else max_x
                max_y = abs(wrist_relative_coords[1]) if abs(wrist_relative_coords[1]) > max_y else max_y

                arr.append(
                    f"{wrist_relative_coords[0]}|{wrist_relative_coords[1]}")

            write_arr.append(arr)

        for row, landmark in enumerate(write_arr):
            for col, coord in enumerate(write_arr[row]):
                string = write_arr[row][col]
                tokens = string.split("|")

                write_arr[row][col] = f"{float(tokens[0]) / float(max_x)}|{float(tokens[1]) / float(max_y)}"

        writer.writerows(write_arr)
        print("Data written")


def get_relative_coord(origin: tuple, point: tuple) -> tuple:
    """
    Returns the coordinate of the second point as if the first point was the origin
    """
    return point[0] - origin[0], point[1] - origin[1]


if __name__ == "__main__":
    main()
