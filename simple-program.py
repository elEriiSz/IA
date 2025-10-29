import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

# 1️⃣ Crear una pequeña red neuronal CNN para predecir (x, y) de la pupila
def build_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(2, activation="sigmoid")
    ])
    return model

# En este ejemplo la red no está entrenada; en práctica deberías entrenarla
# con un dataset de ojos etiquetados (coordenadas de pupila).
# Para la demo, simulamos una red con pesos aleatorios:
model = load_model("pupil_tracker.keras", compile=False)

# 2️⃣ Cargar detector de ojos de OpenCV
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 3️⃣ Captura de video

ip_camera_url = "http://192.168.1.34:4747/video"
cap = cv2.VideoCapture(ip_camera_url)

print("Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (ex, ey, ew, eh) in eyes:
        eye_roi = gray[ey:ey+eh, ex:ex+ew]
        eye_resized = cv2.resize(eye_roi, (64, 64))
        eye_input = eye_resized.reshape(1, 64, 64, 1) / 255.0

        # 4️⃣ Predicción (simulada porque la red no está entrenada)
        pred = model.predict(eye_input, verbose=0)[0]
        px, py = int(pred[0] * ew), int(pred[1] * eh)

        # 5️⃣ Dibujar resultados
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        cv2.circle(frame, (ex + px, ey + py), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"({px},{py})", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Seguimiento de Pupila", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

