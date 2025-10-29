"""
build_pupil_dataset.py

Crea automáticamente un dataset de imágenes de ojos y posiciones de pupila
usando la cámara (local o IP) con MediaPipe Face Mesh (Iris detection).

Guarda imágenes recortadas del ojo izquierdo y derecho, con coordenadas normalizadas
de la pupila (x_norm, y_norm) dentro del recorte.

Controles:
    s - guardar muestra actual (ambos ojos)
    q - salir

Estructura del dataset resultante:
dataset/
 ├── left/
 │    ├── images/
 │    └── labels/
 └── right/
      ├── images/
      └── labels/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------- Configuración ----------------
USE_IP_CAMERA = False
IP_CAMERA_URL = "http://192.168.1.34:4747/video"   # reemplazá con tu URL si usás cámara del celular
SAVE_DIR = "dataset"
EYE_IMG_SIZE = 64
SHOW_PREVIEW = True
SAVE_INTERVAL = 0.5  # segundos mínimos entre guardados

# Crear carpetas necesarias
for side in ["left", "right"]:
    os.makedirs(f"{SAVE_DIR}/{side}/images", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/{side}/labels", exist_ok=True)

# ---------------- MediaPipe setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks de iris y ojo
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380, 381, 382]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

def get_eye_region(landmarks, eye_idx, iris_idx, img_w, img_h, frame):
    """Devuelve recorte del ojo, centro de pupila normalizado dentro del recorte"""
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_idx], dtype=np.float32)
    iris_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in iris_idx], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    pad = int(0.3 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None, None

    # Centro de iris
    iris_center = iris_pts.mean(axis=0)
    cx, cy = iris_center

    # Coordenadas normalizadas dentro del recorte
    x_norm = (cx - x1) / (x2 - x1)
    y_norm = (cy - y1) / (y2 - y1)

    eye_crop = cv2.resize(eye_crop, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    return eye_crop, (x_norm, y_norm)

# ---------------- Captura ----------------
use_ip = input("¿Usar cámara IP del celular? (s/n): ").strip().lower() == "s"
if use_ip:
    USE_IP_CAMERA = True
    ip_url = input(f"Ingresá la URL (ej: {IP_CAMERA_URL}): ").strip()
    if ip_url:
        IP_CAMERA_URL = ip_url

cap = cv2.VideoCapture(IP_CAMERA_URL if USE_IP_CAMERA else 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Presioná 's' para guardar una muestra, 'q' para salir.")

last_save_time = 0
sample_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la cámara.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    img_h, img_w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye_crop, left_label = get_eye_region(landmarks, LEFT_EYE_IDX, LEFT_IRIS_IDX, img_w, img_h, frame)
        right_eye_crop, right_label = get_eye_region(landmarks, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, img_w, img_h, frame)

        if left_eye_crop is not None and right_eye_crop is not None:
            if SHOW_PREVIEW:
                cv2.imshow("Left Eye", left_eye_crop)
                cv2.imshow("Right Eye", right_eye_crop)

            # Dibujar iris
            for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
                lm = landmarks[idx]
                px, py = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        if left_eye_crop is not None and right_eye_crop is not None:
            now = time.time()
            if now - last_save_time > SAVE_INTERVAL:
                idx = sample_counter
                sample_counter += 1

                # Guardar ojo izquierdo
                cv2.imwrite(f"{SAVE_DIR}/left/images/eye_{idx:05d}.png", left_eye_crop)
                with open(f"{SAVE_DIR}/left/labels/eye_{idx:05d}.txt", "w") as f:
                    f.write(f"{left_label[0]:.5f} {left_label[1]:.5f}")

                # Guardar ojo derecho
                cv2.imwrite(f"{SAVE_DIR}/right/images/eye_{idx:05d}.png", right_eye_crop)
                with open(f"{SAVE_DIR}/right/labels/eye_{idx:05d}.txt", "w") as f:
                    f.write(f"{right_label[0]:.5f} {right_label[1]:.5f}")

                print(f"Muestras guardadas #{idx}: Izq={left_label}, Der={right_label}")
                last_save_time = now

cap.release()
cv2.destroyAllWindows()

