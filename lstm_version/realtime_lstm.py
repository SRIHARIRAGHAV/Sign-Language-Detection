import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model("models/lstm_model.h5")

# Load labels
labels = os.listdir("data")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
sequence_length = 30
current_word = ""

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        frame_keypoints = []
        for lm in hand_landmarks.landmark:
            frame_keypoints.extend([lm.x, lm.y, lm.z])

        sequence.append(frame_keypoints)

        if len(sequence) > sequence_length:
            sequence.pop(0)

        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)
            predicted_label = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            if confidence > 0.8:
                current_word = predicted_label

    cv2.putText(frame, f"Word: {current_word}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("LSTM Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
