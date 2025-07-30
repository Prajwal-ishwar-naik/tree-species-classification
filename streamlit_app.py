import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Page Config & Styling ------------------- #
st.set_page_config(page_title="🌳 Tree Species Classifier", layout="centered")

st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        text-align: center;
        color: #2e7d32;
        font-weight: bold;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }
    .result {
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Title ------------------- #
st.markdown("<div class='main-title'>🌿 Tree Species Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Upload a tree image (leaf, bark, etc.) and the model will predict its species.</div>", unsafe_allow_html=True)

# ------------------- Load Model ------------------- #
@st.cache_resource
def load_trained_model():
    model = load_model("basic_cnn_tree_species.h5")
    return model

model = load_trained_model()
st.success("✅ Model loaded: basic_cnn_tree_species.h5")

# ------------------- Class Labels ------------------- #
class_labels = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa', 'coconut',
    'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango', 'motichanoti',
    'neem', 'nilgiri', 'other', 'pilikaren', 'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal',
    'sonmahor', 'sugarcane', 'vad'
]  # ✅ 30 unique class names — no duplicates!

# ------------------- File Uploader ------------------- #
uploaded_file = st.file_uploader("📤 Upload a Tree Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing the image..."):
        # Preprocess
        img = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # ------------------- Result Display ------------------- #
    st.markdown(f"<div class='result'>🌳 <strong>Predicted Species:</strong> <code>{predicted_class}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>🔎 <strong>Confidence:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)

    # ------------------- Optional Warning ------------------- #
    if confidence < 50:
        st.warning("⚠️ The model is not confident. Try using a clearer or different image.")

    # ------------------- Probability Chart ------------------- #
    top_5_idx = prediction.argsort()[-5:][::-1]
    top_5_labels = [class_labels[i] for i in top_5_idx]
    top_5_values = prediction[top_5_idx] * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(top_5_labels[::-1], top_5_values[::-1], color='#66bb6a')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top 5 Predicted Species')
    ax.bar_label(bars, fmt='%.2f%%')
    st.pyplot(fig)
