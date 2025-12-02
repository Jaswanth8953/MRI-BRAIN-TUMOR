import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

MODEL_CONFIG = {
    "EfficientNetB0": {
        "url": "https://drive.google.com/uc?id=1kfLjAHoGbYNBg12gU-vr9rCsnVX-kCFG",
        "filename": "efficientnet_finetuned.h5"
    },
    "MobileNetV3": {
        "url": "https://drive.google.com/uc?id=1ARZryz9H5Bc832iRs2DiPyEjBL5BuNKR",
        "filename": "mobilenet_finetuned.h5"
    }
}

# ---------------------------------------------------------
# DOWNLOAD MODEL IF NEEDED
# ---------------------------------------------------------
def download_model_if_needed(model_name):
    cfg = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    path = f"models/{cfg['filename']}"

    if not os.path.exists(path):
        with st.spinner(f"Downloading {model_name} model…"):
            gdown.download(cfg["url"], path, quiet=False)

    return path

# ---------------------------------------------------------
# SAFE MODEL LOADER (NO DISABLE EAGER EXECUTION)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    model_path = download_model_if_needed(model_name)

    try:
        # Try normal loading (EfficientNet works)
        return tf.keras.models.load_model(model_path, compile=False)

    except Exception:
        st.warning(f"Standard load failed for {model_name}. Safe-loading MobileNetV3…")

        try:
            # Rebuild MobileNetV3Small architecture
            model = tf.keras.applications.MobileNetV3Small(
                input_shape=(224,224,3),
                include_top=True,
                weights=None,
                classes=4
            )
            model.load_weights(model_path)  # Load only weights
            return model

        except Exception as e:
            st.error(f"Failed to load MobileNetV3: {e}")
            raise e

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.markdown("<h1 style='text-align:center;color:#3B82F6;'>🧠 Brain MRI Tumor Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["EfficientNetB0", "MobileNetV3"],
    index=1
)

st.sidebar.info("Upload MRI and click Run Prediction.")

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

uploaded = st.file_uploader("Upload MRI Here", type=["jpg","jpeg","png"])

with col1:
    if uploaded:
        arr, img_display = preprocess_image(uploaded)
        st.image(img_display, caption="Uploaded MRI", use_column_width=True)
    else:
        st.info("Please upload an MRI image.")

with col2:
    if uploaded:
        if st.button("Run Prediction", use_container_width=True):
            try:
                model = load_model(model_choice)

                with st.spinner("Analyzing MRI…"):
                    preds = model.predict(arr)[0]

                predicted_class = CLASS_NAMES[np.argmax(preds)]
                confidence = float(np.max(preds) * 100)

                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

                # Probability table
                st.subheader("📊 Probability Distribution")
                prob_df = {
                    "Tumor Type": CLASS_NAMES,
                    "Probability (%)": (preds * 100).round(2)
                }
                st.table(prob_df)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ---------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------

