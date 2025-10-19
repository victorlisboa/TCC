from ultralytics import YOLO
import cv2

# modelo pré-treinado em mãos (precisa de dataset de mãos)
model = YOLO("hand_detection/yolo_models/lewiswatsonyolov8x-tuned-hand-gestures.pt")  # substitua por um modelo de mãos

cap = cv2.VideoCapture("/mnt/d/videos_palavras/2/pessoa3video1-02.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results[0].boxes.xyxy:  # bounding boxes [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, r.tolist())
        hand_crop = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
