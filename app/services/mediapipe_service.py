
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    return [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]
