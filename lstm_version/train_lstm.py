import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = "data"

X = []
y = []

labels = os.listdir(data_path)

for label in labels:
    label_path = os.path.join(data_path, label)
    for file in os.listdir(label_path):
        X.append(np.load(os.path.join(label_path, file)))
        y.append(label)

X = np.array(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2
)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 63)))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")

print("Model training completed and saved.")
