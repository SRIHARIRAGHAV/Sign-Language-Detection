from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",
    epochs=50,
    imgsz=224,
    batch=16,
    device=0  
)
