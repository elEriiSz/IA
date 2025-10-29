"""
train_pupil_tracker.py

Entrena una red neuronal profunda para estimar la posición de la pupila
a partir del dataset generado automáticamente por auto_pupil_dataset.py.

Estructura esperada del dataset:
dataset/
 ├── left/
 │    ├── images/
 │    └── labels/
 └── right/
      ├── images/
      └── labels/
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
IMG_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 32
MODEL_PATH = "pupil_tracker.keras"

# ---------------- FUNCIONES ----------------
def load_eye_data(path):
    """Carga imágenes y coordenadas desde un subdirectorio (left o right)."""
    images_dir = os.path.join(path, "images")
    labels_dir = os.path.join(path, "labels")

    X, y = [], []

    for file in sorted(os.listdir(images_dir)):
        if not file.endswith(".png"):
            continue
        img_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, file.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=-1)  # de (64,64) → (64,64,1)
        img = img.astype("float32") / 255.0

        with open(label_path, "r") as f:
            x, y_coord = map(float, f.readline().split())

        X.append(img)
        y.append([x, y_coord])

    return np.array(X), np.array(y)

# ---------------- CARGA DE DATOS ----------------
print("Cargando dataset...")

X_left, y_left = load_eye_data(os.path.join(DATASET_DIR, "left"))
X_right, y_right = load_eye_data(os.path.join(DATASET_DIR, "right"))

X = np.concatenate([X_left, X_right], axis=0)
y = np.concatenate([y_left, y_right], axis=0)

print(f"Total de muestras: {len(X)}")

# ---------------- DIVISIÓN TRAIN/VAL ----------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ---------------- MODELO DEEP LEARNING ----------------
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(2, activation="sigmoid")  # salida (x, y) normalizada entre 0 y 1
    ])
    return model

model = build_model()
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

model.summary()

# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)

# ---------------- ENTRENAMIENTO ----------------
print("Entrenando modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, lr_reducer, early_stop],
    verbose=1
)

# ---------------- VISUALIZACIÓN ----------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss (MSE)")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["mae"], label="train_mae")
plt.plot(history.history["val_mae"], label="val_mae")
plt.title("Error Absoluto Medio")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n✅ Modelo entrenado y guardado como: {MODEL_PATH}")

