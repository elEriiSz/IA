import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

# 📁 Directorios del dataset
img_dir = "dataset/images"
label_dir = "dataset/labels"

# 🧠 Parámetros
img_size = 32

# 🧩 Cargar imágenes y etiquetas
X, y = [], []
files = sorted(os.listdir(img_dir))

print("📦 Cargando dataset...")
for fname in tqdm(files):
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))

    # Leer imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # normalizar
    X.append(img.reshape(img_size, img_size, 1))

    # Leer etiqueta
    with open(label_path, "r") as f:
        coords = list(map(float, f.readline().strip().split()))
    y.append(coords)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"✅ Dataset cargado: {len(X)} imágenes")

# 🧮 División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 🧠 Definir el modelo CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(2, activation='sigmoid')  # (x, y) normalizados
])

# ⚙️ Compilar modelo
model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])

# 💾 Guardar el mejor modelo automáticamente
checkpoint = ModelCheckpoint("pupil_tracker.h5", monitor="val_loss", save_best_only=True, verbose=1)

# 🚀 Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

print("✅ Entrenamiento finalizado. Modelo guardado como pupil_tracker.h5")

