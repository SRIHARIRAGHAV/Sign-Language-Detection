
# Sign Language Detection System – Backend

This repository contains the **backend implementation** of a real-time Sign Language Detection System designed to assist deaf and mute individuals by converting hand gestures into text and speech.

## 🚀 Features
- Real-time hand gesture detection
- MediaPipe-based hand landmark extraction
- Advanced CNN (InceptionV3) + YOLOv8 for gesture classification
- Text-to-Speech conversion
- REST API built with FastAPI
- Modular, scalable, and deployment-ready

## 🏗️ System Architecture
1. Data Acquisition (Webcam / Image Input)
2. Preprocessing (MediaPipe landmarks)
3. Feature Extraction (InceptionV3)
4. Gesture Classification (YOLOv8)
5. Text-to-Speech Conversion
6. User Interface Integration

## 📁 Project Structure
```
Sign-Language-Detection-Backend/
├── app/
│   ├── main.py
│   ├── services/
│   │   ├── mediapipe_service.py
│   │   ├── preprocessing.py
│   │   ├── gesture_classifier.py
│   │   └── tts_service.py
│   ├── models/
│   └── utils/
├── outputs/audio/
├── requirements.txt
├── README.md
```

## ▶️ Run the Project
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open API Docs:
http://127.0.0.1:8000/docs

## 🎓 Academic Use
This project is suitable for:
- Final Year Engineering Projects
- AI / Computer Vision Demonstrations
- Research Prototypes

## 📌 Future Enhancements
- Mobile app integration
- Multilingual sign support
- Continuous gesture recognition
- Cloud deployment

---
**Developed for academic and assistive technology purposes.**
