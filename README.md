# Hand Gesture Detection

Not really sure what I was trying to accomplish here but it turned out better than I thought. 
The program takes webcam video as input, marks the gestures recognized from each hand shown in the video then returns the edited image.

## How It Works

### Data Collection
- Hands and their respective landmarks are collected through [Google's MediaPipe API](https://mediapipe.dev/)
- Data collection was inspired by [this](https://github.com/kinivi/hand-gesture-recognition-mediapipe) gesture detection library
  - For each landmark:
    - Convert the normalized x,y coordinates into pixel coordinates
    - Convert the pixel coordinaates into a relative coordinate to the palm
    - Normalize the x,y values respectively according to the highest overall value for the hand
- Write the data to a CSV file for each gesture

### NN Training
- Concatenate separate CSV files into a master csv file (to be fair we could also append the data)
- Read in the CSV into a `pandas.DataFrame`
- Assign each label an integer indentifier using a plain Python `dict()` (there may also be a way to build the values into the tf model but I'm not quite sure how to atm)
- Compile a Sequential ML model (`tf.keras.models.Sequential`)
  - The output dense layer should have `activation="softmax"` to get probabilities of each gesture and `units=NUM_RECOGNIZED_GESTURES`
  - Loss function during fitting should be `sparse_categorical_crossentropy` (to the best of my knowledge this is used for non-binary classification?)
- The resulting model should be able to classify a hand gesture provided an array of values processed in the same way described in Data Collection


## Dependencies
- [MediaPipe](https://mediapipe.dev/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV (python)](https://pypi.org/project/opencv-python/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
