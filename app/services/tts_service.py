
from gtts import gTTS
import uuid, os

def text_to_speech(text):
    os.makedirs("outputs/audio", exist_ok=True)
    path = f"outputs/audio/{uuid.uuid4()}.mp3"
    gTTS(text=text, lang='en').save(path)
    return path
