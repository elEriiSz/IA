import cv2
import numpy as np
import mediapipe as mp

# Inicializar mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # incluye los landmarks del iris
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Crear un lienzo (canvas) negro del mismo tamaño que el video
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Color del dibujo
color = (0, 255, 0)
thickness = 2

prev_pos = None  # posición anterior de la pupila

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Coordenadas de landmarks del ojo derecho (ejemplo)
        iris_right_idx = [474, 475, 476, 477]  # landmarks del iris derecho
        iris_points = []

        for idx in iris_right_idx:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            iris_points.append((x, y))

        # Centro del iris
        iris_points = np.array(iris_points)
        cx, cy = np.mean(iris_points, axis=0).astype(int)

        # Dibujar punto en la pupila
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Dibujar en el canvas si hay posición previa
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, (cx, cy), color, thickness)

        prev_pos = (cx, cy)

    # Superponer el dibujo sobre la imagen
    output = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    cv2.imshow("Eye Paint", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas[:] = 0  # limpiar
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

