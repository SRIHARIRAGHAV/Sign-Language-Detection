from ultralytics import YOLO
import cv2

model = YOLO("app/models/yolov8_model.pt")

def classify_gesture(frame, hand_bbox):
    x1, y1, x2, y2 = hand_bbox
    hand_img = frame[y1:y2, x1:x2]

    if hand_img.size == 0:
        return "Unknown"

    hand_img = cv2.resize(hand_img, (224, 224))

    result = model.predict(hand_img, verbose=False)

    class_id = int(result[0].probs.top1)
    gesture_label = result[0].names[class_id]

    return gesture_label
