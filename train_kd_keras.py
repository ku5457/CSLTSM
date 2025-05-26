# train_kd_keras.py
# Train SMCNN and SMTSNN using knowledge distillation in Keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from smcnn_smtsnn_keras import build_smcnn, build_smtsnn

# ---------------------------
# Load soft labels & original labels
# ---------------------------
x_cnn = np.random.rand(160, 80, 96, 1).astype(np.float32)
x_tsnn = np.random.rand(160, 30, 512).astype(np.float32)
soft_cnn = np.load("teacher_logits_cnn.npy")
soft_tsnn = np.load("teacher_logits_tsnn.npy")
hard_labels = np.load("train_labels.npy")

# ---------------------------
# Distillation Parameters
# ---------------------------
temperature = 4.0
alpha = 0.7

# ---------------------------
# Custom Distillation Loss
# ---------------------------
def distillation_loss(y_true, y_pred, teacher_soft, temperature, alpha):
    kl = KLDivergence()(tf.nn.softmax(teacher_soft / temperature), tf.nn.log_softmax(y_pred / temperature)) * (temperature ** 2)
    ce = SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    return alpha * kl + (1 - alpha) * ce

# ---------------------------
# Build Models
# ---------------------------
smcnn = build_smcnn()
smtsnn = build_smtsnn()
optimizer = Adam()

# ---------------------------
# Training Loop
# ---------------------------
epochs = 10
batch_size = 32

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, len(x_cnn), batch_size):
        xb_cnn = x_cnn[i:i+batch_size]
        xb_tsnn = x_tsnn[i:i+batch_size]
        yb_soft_cnn = soft_cnn[i:i+batch_size]
        yb_soft_tsnn = soft_tsnn[i:i+batch_size]
        yb_true = hard_labels[i:i+batch_size]

        with tf.GradientTape() as tape1:
            logits_cnn = smcnn(xb_cnn, training=True)
            loss1 = distillation_loss(yb_true, logits_cnn, yb_soft_cnn, temperature, alpha)

        grads1 = tape1.gradient(loss1, smcnn.trainable_variables)
        optimizer.apply_gradients(zip(grads1, smcnn.trainable_variables))

        with tf.GradientTape() as tape2:
            logits_tsnn = smtsnn(xb_tsnn, training=True)
            loss2 = distillation_loss(yb_true, logits_tsnn, yb_soft_tsnn, temperature, alpha)

        grads2 = tape2.gradient(loss2, smtsnn.trainable_variables)
        optimizer.apply_gradients(zip(grads2, smtsnn.trainable_variables))

    print(f"CNN KD Loss: {loss1.numpy():.4f}, TSNN KD Loss: {loss2.numpy():.4f}")

# ---------------------------
# Save models if needed
# ---------------------------
smcnn.save("smcnn_kd.h5")
smtsnn.save("smtsnn_kd.h5")
print("KD training completed and models saved.")
