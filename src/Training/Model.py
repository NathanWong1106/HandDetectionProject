# https://www.tensorflow.org/tutorials/load_data/csv
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import mediapipe as mp
import keras
from src.Util.Detector import Detector
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from DatasetConstructor import get_normalized_relative_landmarks


def main():
    dataframe = pd.read_csv('./Data/concat.csv')

    features = dataframe.copy()
    labels = features.pop('gesture')
    labels = LabelEncoder().fit_transform(labels.array)
    features = np.array(features)

    print(labels)

    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # hidden layers
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # softmax activation returns probabilities of results
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(features, labels, epochs=5, batch_size=1)

    # TODO: Compile and save model
    # detector = Detector()
    # img = cv2.imread("./Data/open_palm.jpg")
    # hand_landmarks = detector.get_landmarks(img).multi_hand_landmarks
    #
    # for hand in hand_landmarks:
    #     arr = get_normalized_relative_landmarks(img, hand)
    #     arr = np.array(arr).reshape((1, 42))
    #     predict = model.predict(arr).argmax()
    #     print(predict)


if __name__ == "__main__":
    main()
