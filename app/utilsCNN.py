import cv2 as cv
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.utils import np_utils

np.set_printoptions(threshold=np.inf)

# cnn_model = tf.keras.models.load_model('../static/cnnModel')
cnn_model = tf.keras.models.load_model('/code/app/cnnModel_3class')

# ? define hyperparameters
IMG_SIZE = 224
# IMG_SIZE = 500
BATCH_SIZE = 64
EPOCHS = 100

MAX_SEQ_LENGTH = 500
# MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path):
    class_vocab = ['BodyWeightSquats', 'PullUps', 'PushUps']
    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = cnn_model.predict([frame_features, frame_mask])[0]

    maxProb = 0
    index = None
    for i in np.argsort(probabilities)[::-1]:
      probCalc = probabilities[i] * 100
      if(probCalc > maxProb):
        maxProb = probCalc
        index = i
      print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    # return frames # ? should not be returning frames?
    return class_vocab[index]