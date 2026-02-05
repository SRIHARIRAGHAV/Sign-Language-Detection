
import numpy as np

def preprocess_landmarks(landmarks):
    data = np.array(landmarks).flatten()
    return data / np.max(data)
