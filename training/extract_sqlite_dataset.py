import sqlite3

db_path = "sign-language-recognition.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in database:")
for t in tables:
    print(t[0])

conn.close()

import sqlite3
import os
import cv2
import numpy as np

db_path = "sign-language-recognition.sqlite"
output_dir = "training/dataset/train"

os.makedirs(output_dir, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# CHANGE table & column names after inspection
query = "SELECT image, label FROM gestures"
cursor.execute(query)

rows = cursor.fetchall()

for i, (img_blob, label) in enumerate(rows):
    label_dir = os.path.join(output_dir, label.lower())
    os.makedirs(label_dir, exist_ok=True)

    # Convert blob to image
    img_array = np.frombuffer(img_blob, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is not None:
        cv2.imwrite(f"{label_dir}/{label}_{i}.jpg", img)

conn.close()
print("Dataset extraction completed.")
