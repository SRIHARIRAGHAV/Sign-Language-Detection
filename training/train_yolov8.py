from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# Train using your dataset
model.train(
    data="training/dataset",
    epochs=30,
    imgsz=224,
    batch=16
)
