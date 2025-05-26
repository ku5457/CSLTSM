# csltsm_net_keras.py
# Keras implementation of CSLTSM-Net (SMCNN + SMTSNN + fusion layer)

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

# ---------------------------
# CSLTSM-Net: Fusion Model
# ---------------------------
def build_csltsm_net():
    cnn_input = Input(shape=(80, 96, 1), name="cnn_input")
    tsnn_input = Input(shape=(30, 512), name="tsnn_input")

    smcnn = build_smcnn()
    smtsnn = build_smtsnn()

    cnn_out = smcnn(cnn_input)
    tsnn_out = smtsnn(tsnn_input)

    fused = layers.Concatenate()([cnn_out, tsnn_out])
    final_out = layers.Dense(2, name="fusion_output")(fused)

    model = models.Model(inputs=[cnn_input, tsnn_input], outputs=final_out, name="CSLTSM_Net")
    return model

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    model = build_csltsm_net()
    model.summary()
