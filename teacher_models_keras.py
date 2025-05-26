# teacher_models_keras.py
# Keras implementation of TMCNN and TMTSNN (Teacher Models)

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ---------------------------
# TMCNN: Teacher CNN Model
# ---------------------------
def build_tmcnn(input_shape=(80, 96, 1)):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2)(x)
    return models.Model(inputs, outputs, name="TMCNN")

# ---------------------------
# TMTSNN: Teacher Time-Series Model
# ---------------------------
def build_tmtsnn(input_shape=(30, 512)):
    inputs = Input(shape=input_shape)
    x = layers.GRU(64)(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(2)(x)
    return models.Model(inputs, outputs, name="TMTSNN")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    tmcnn = build_tmcnn()
    tmtsnn = build_tmtsnn()
    print("TMCNN Summary:")
    tmcnn.summary()
    print("TMTSNN Summary:")
    tmtsnn.summary()

