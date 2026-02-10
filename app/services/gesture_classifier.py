import cv2
from ultralytics import YOLO

# Load trained YOLOv8 classification model
model = YOLO("app/models/yolov8_model.pt")

def classify_gesture(frame, hand_bbox):
    """
    frame: original image
    hand_bbox: (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = hand_bbox
    hand_img = frame[y1:y2, x1:x2]

    if hand_img.size == 0:
        return "Unknown"

    # Resize for YOLOv8 classification
    hand_img = cv2.resize(hand_img, (224, 224))

    # YOLOv8 prediction
    result = model.predict(hand_img, verbose=False)

    class_id = int(result[0].probs.top1)
    gesture_label = result[0].names[class_id]

    return gesture_label
