# evaluate_keras.py
# Evaluation script for SMCNN, SMTSNN, CSLTSM-Net (Keras version)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from smcnn_smtsnn_keras import build_smcnn, build_smtsnn
from csltsm_net_keras import build_csltsm_net

# ---------------------------
# Load test data (simulate or replace)
# ---------------------------
x_cnn = np.random.rand(40, 80, 96, 1).astype(np.float32)
x_tsnn = np.random.rand(40, 30, 512).astype(np.float32)
y_true = np.random.randint(0, 2, size=(40,))

# ---------------------------
# Load trained models
# ---------------------------
smcnn = tf.keras.models.load_model("smcnn_kd.h5")
smtsnn = tf.keras.models.load_model("smtsnn_kd.h5")
csltsm = build_csltsm_net()
csltsm.get_layer("SMCNN").set_weights(smcnn.get_weights())
csltsm.get_layer("SMTSNN").set_weights(smtsnn.get_weights())

# ---------------------------
# Evaluate predictions
# ---------------------------
def evaluate_model(model, name, inputs):
    preds = model.predict(inputs)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Evaluation Results:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

# Run evaluations
evaluate_model(smcnn, "SMCNN", x_cnn)
evaluate_model(smtsnn, "SMTSNN", x_tsnn)
evaluate_model(csltsm, "CSLTSM-Net", [x_cnn, x_tsnn])
