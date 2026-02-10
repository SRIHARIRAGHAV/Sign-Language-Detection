from ultralytics import YOLO

model = YOLO("app/models/yolov8_model.pt")

def classify_gesture(landmarks):
    # inference logic
    pass
