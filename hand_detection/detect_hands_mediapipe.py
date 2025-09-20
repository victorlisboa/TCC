import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# inicializa detector
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)

# abre vídeo
cap = cv2.VideoCapture("/mnt/d/videos_palavras/1/pessoa1video1-01.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # converte para RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # processa frame
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # pega pontos da mão
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]

            # bounding box
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # recorte da mão
            hand_crop = frame[y_min:y_max, x_min:x_max]

            # mostra para debug
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

    cv2.imshow("Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
