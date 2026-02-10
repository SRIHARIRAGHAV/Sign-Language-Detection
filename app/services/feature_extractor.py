import tensorflow as tf
import numpy as np

model = tf.keras.applications.InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

def extract_features(image):
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features
