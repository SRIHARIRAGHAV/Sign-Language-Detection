from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np
from services.mediapipe_service import detect_hand
from services.gesture_classifier import classify_gesture
from services.tts_service import text_to_speech

app = FastAPI(title="Sign Language Detection Backend")

@app.post("/detect")
async def detect_gesture(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    bbox = detect_hand(frame)
    if bbox is None:
        return {"error": "No hand detected"}

    gesture = classify_gesture(frame, bbox)
    audio_path = text_to_speech(gesture)

    return {
        "gesture": gesture,
        "audio": audio_path
    }
