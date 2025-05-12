import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model(r'C:\Users\FINRISE\Desktop\cat_dog_classifier\cat_dog_classifier_model.h5')

# Class names (adjust based on your dataset folders)
class_names = ['Cat', 'Dog']

st.title("🐶🐱 Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize((180, 180))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict
    prediction = model.predict(image_array)
    label = class_names[int(prediction[0][0] > 0.5)]

    st.markdown(f"### Prediction: `{label}`")
