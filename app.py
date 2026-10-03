import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("cats_dogs_efficientnetb0.keras")

st.title("🐱 Cats vs Dogs Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_names = ["cats", "dogs"]
    predicted_name = class_names[predicted_class]

    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_name}")
    st.write(f"Confidence: {confidence:.2f}%")