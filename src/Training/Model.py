# https://www.tensorflow.org/tutorials/load_data/csv
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from src.Training.DatasetConstructor import get_normalized_relative_landmarks
from src.Training.DataBuilder import OUTPUT_FILE_PATH

MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "Trained", "trained_model")

# Format: Data Name: TF Label, TF Label: Display Name
GESTURES = {
    "Open_Palm": 0, 0: "Open Palm",
    "Fist": 1, 1: "Fist",
    "Pointing": 2, 2: "Pointing",
    "Spooder-Man": 3, 3: "Spooder-Man",
    "Peace": 4, 4: "Peace"
}

NUM_RECOGNIZED_GESTURES = len(GESTURES) / 2
EPOCHS = 5
BATCH_SIZE = 64


class GestureModel:

    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def get_gesture_prediction(self, img, hand) -> str:
        arr = get_normalized_relative_landmarks(img, hand)
        arr = np.array(arr).reshape((1, 42))
        predict = self.model.predict(arr).argmax()
        return GESTURES[predict]

    def __get_highest_probability(self, arr):
        pass


def main():
    dataframe = pd.read_csv(OUTPUT_FILE_PATH)

    features = dataframe.copy()
    labels = features.pop('gesture')
    features = np.array(features)

    for index, val in enumerate(labels):
        labels[index] = GESTURES[val]
    labels = labels.to_numpy().astype(int)

    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Dense(128, activation="relu"),

        # Dropout of certain nodes protects against overfitting
        tf.keras.layers.Dropout(0.5),

        # hidden layers
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # softmax activation returns probabilities of results
        tf.keras.layers.Dense(NUM_RECOGNIZED_GESTURES, activation="softmax")
    ])

    # Sparse categorical crossentropy is used for non-binary classification
    # (to the best of my very limited knowledge lol)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(features, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

    save = True if input("Save Model (yes/no): ") == "yes" else False

    if save:
        model.save(MODEL_PATH, overwrite=True)


if __name__ == "__main__":
    main()
