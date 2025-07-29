import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Set page config
st.set_page_config(page_title="🌳 Tree Species Classifier", layout="centered")

st.title("🌿 Tree Species Classification using CNN")
st.write("Upload a tree image (leaf, bark, etc.) and the model will predict its species.")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("improved_cnn_model.h5")  # Make sure this file is in the same directory
    return model

model = load_trained_model()

# Define your class labels (order must match the training generator)
class_labels = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa', 'coconut',
    'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango', 'motichanoti',
    'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal',
    'sonmahor', 'sugarcane', 'vad', 'gunda'
]

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='📷 Uploaded Image', use_column_width=True)

    # Preprocess image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.markdown("### 🧠 Predicted Species:")
    st.success(predicted_class)