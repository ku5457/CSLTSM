# train_teacher_keras.py
# Keras training script for TMCNN and TMTSNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from teacher_models_keras import build_tmcnn, build_tmtsnn

# ---------------------------
# Simulated Data (Replace with real preprocessed data)
# ---------------------------
x_cnn = np.random.rand(200, 80, 96, 1).astype(np.float32)
x_tsnn = np.random.rand(200, 30, 512).astype(np.float32)
y = np.random.randint(0, 2, size=(200,)).astype(np.int32)

# Split into train/test
train_x_cnn, test_x_cnn = x_cnn[:160], x_cnn[160:]
train_x_tsnn, test_x_tsnn = x_tsnn[:160], x_tsnn[160:]
train_y, test_y = y[:160], y[160:]

# ---------------------------
# Build and Compile Models
# ---------------------------
tmcnn = build_tmcnn()
tmtsnn = build_tmtsnn()

loss_fn = SparseCategoricalCrossentropy(from_logits=True)
tmcnn.compile(optimizer=Adam(), loss=loss_fn, metrics=['accuracy'])
tmtsnn.compile(optimizer=RMSprop(), loss=loss_fn, metrics=['accuracy'])

# ---------------------------
# Train Models
# ---------------------------
tmcnn.fit(train_x_cnn, train_y, epochs=10, batch_size=32, validation_data=(test_x_cnn, test_y))
tmtsnn.fit(train_x_tsnn, train_y, epochs=10, batch_size=32, validation_data=(test_x_tsnn, test_y))

# ---------------------------
# Save soft label outputs for KD training
# ---------------------------
logits_cnn = tmcnn.predict(train_x_cnn)
logits_tsnn = tmtsnn.predict(train_x_tsnn)

np.save("teacher_logits_cnn.npy", logits_cnn)
np.save("teacher_logits_tsnn.npy", logits_tsnn)
np.save("train_labels.npy", train_y)

print("Teacher models trained and soft labels saved.")
