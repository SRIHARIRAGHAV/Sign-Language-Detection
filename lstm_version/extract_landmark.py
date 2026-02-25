import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Paths
video_root = "selected_files"
output_root = "data"

# Create output folders if not exist
os.makedirs(output_root, exist_ok=True)

sequence_length = 30

for word in os.listdir(video_root):
    word_path = os.path.join(video_root, word)
    output_word_path = os.path.join(output_root, word)

    os.makedirs(output_word_path, exist_ok=True)

    file_count = 0

    for video_file in os.listdir(word_path):
        video_path = os.path.join(word_path, video_file)

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while cap.isOpened():
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

            # Stop after collecting 30 frames
            if len(sequence) == sequence_length:
                break

        cap.release()

        # Save only if we collected full sequence
        if len(sequence) == sequence_length:
            np.save(os.path.join(output_word_path, f"{file_count}.npy"), np.array(sequence))
            file_count += 1

    print(f"Finished processing word: {word}")

print("Landmark extraction completed.")
