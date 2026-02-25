import cv2
from ultralytics import YOLO

model = YOLO("app/models/yolov8_model.pt")

cap = cv2.VideoCapture(0)

current_message = ""
last_label = ""
stable_count = 0
STABLE_THRESHOLD = 25  # frames (~1 sec)

word_map = {
    "A": "HELLO",
    "B": "THANK YOU",
    "C": "PLEASE",
    "I": "I LOVE YOU",
    "Y": "YES",
    "N": "NO"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)

    if results[0].probs is not None:
        class_id = int(results[0].probs.top1)
        label = results[0].names[class_id]

        # Check if same label continues
        if label == last_label:
            stable_count += 1
        else:
            stable_count = 0
            last_label = label

        # If stable enough → replace message
        if stable_count == STABLE_THRESHOLD:
            current_message = word_map.get(label, label)
            stable_count = 0

        cv2.putText(frame, f"Current: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Message: {current_message}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
