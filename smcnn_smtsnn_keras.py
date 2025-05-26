# smcnn_smtsnn_keras.py
# Keras implementation of student models: SMCNN and SMTSNN

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ---------------------------
# SMCNN: Student CNN Branch
# ---------------------------
def build_smcnn(input_shape=(80, 96, 1)):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(2, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(16, 16))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(2)(x)
    return models.Model(inputs, outputs, name="SMCNN")

# ---------------------------
# SMTSNN: Student Time-Series Branch
# ---------------------------
def build_smtsnn(input_shape=(30, 512)):
    inputs = Input(shape=input_shape)
    x = layers.GRU(8)(inputs)
    outputs = layers.Dense(2)(x)
    return models.Model(inputs, outputs, name="SMTSNN")
