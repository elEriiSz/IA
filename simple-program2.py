"""
pupil_tracker_live_rgb.py

Usa un modelo entrenado (pupil_tracker_rgb.keras) para rastrear la pupila
en tiempo real con la cámara (webcam o IP del celular).
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract

# ---------------- CONFIG ----------------
MODEL_PATH = "pupil_tracker.keras"
USE_IP_CAMERA = False
IP_CAMERA_URL = "http://192.168.1.34:4747/video"  # si usás cámara del celular
IMG_SIZE = 64

# ---------------- CARGAR MODELO ----------------
print("Cargando modelo...")
model = load_model(MODEL_PATH, compile=False)
print("Modelo cargado correctamente.")

# ---------------- FACE MESH ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks de ojos e iris
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380, 381, 382]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

def shrink_eye_crop(x1, y1, x2, y2, scale=0.8):
    """
    scale < 1 → recorte más pequeño (más cerca del centro)
    scale = 1 → igual tamaño
    scale > 1 → más grande (más margen)
    """
    w = x2 - x1
    h = y2 - y1

    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * scale
    new_h = h * scale

    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    return new_x1, new_y1, new_x2, new_y2


def get_eye_crop(landmarks, eye_idx, img_w, img_h, frame):
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_idx], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    pad = int(0.3 * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)

    x1,y1,x2,y2 = shrink_eye_crop(x1,y1,x2,y2,scale=0.7)

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None, (x1, y1, x2, y2)
    eye_crop = cv2.resize(eye_crop, (IMG_SIZE, IMG_SIZE))
    eye_crop = eye_crop.astype("float32") / 255.0
    return eye_crop, (x1, y1, x2, y2)

# ---------------- CAPTURA ----------------
use_ip = input("¿Usar cámara IP del celular? (s/n): ").strip().lower() == "s"
if use_ip:
    USE_IP_CAMERA = True

cap = cv2.VideoCapture(IP_CAMERA_URL if USE_IP_CAMERA else 0)

print("Iniciando rastreo de pupila... Presioná 'q' para salir.")


# --------------- LIENZO ------------------

# Crear un lienzo (canvas) negro del mismo tamaño que el video
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Color del dibujo
color = (0, 255, 0)
thickness = 2

prev_pos = None  # posición anterior de la pupila
i = 0



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    img_h, img_w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for (eye_idx, iris_idx, color) in [
#            (LEFT_EYE_IDX, LEFT_IRIS_IDX, (0,255,0)),
            (RIGHT_EYE_IDX, RIGHT_IRIS_IDX, (255,0,0))
        ]:
            eye_crop, (x1, y1, x2, y2) = get_eye_crop(landmarks, eye_idx, img_w, img_h, frame)
            if eye_crop is None:
                continue

            pred = model.predict(np.expand_dims(eye_crop, axis=0), verbose=0)[0]
            px = int(x1 + pred[0] * (x2 - x1))
            py = int(y1 + pred[1] * (y2 - y1))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
#            print(((px-x1)/(x2-x1),(py-y1)/(y2-y1)))
            
            # Dibujar en el canvas si hay posición previa
            if prev_pos is not None:
                cv2.line(canvas, prev_pos, (px, py), color, thickness)

            prev_pos = (px, py)

    output = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)
    cv2.imshow("Pupil Tracker (RGB)", frame)
    cv2.imshow("Eye Paint", output)
    key =cv2.waitKey(1) & 0xFF  
    if key == ord("q"):

        break
    elif key == ord('c'):
        canvas[:] = 0
        pass
    elif key == ord('p'):
        drawing = canvas.copy()
        cv2.imwrite("output/eye_"+str(i)+".png", drawing)
        img = cv2.imread("output/eye_"+str(i)+".png")
        img = cv2.resize(img, (400, 450))
        cv2.imshow("Image", img)
        text = pytesseract.image_to_string(img)
        print(text)

        canvas[:] = 0
        i = i +1

        

cap.release()
cv2.destroyAllWindows()

