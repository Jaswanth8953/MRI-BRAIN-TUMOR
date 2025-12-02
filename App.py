import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

MODEL_CONFIG = {
    "EfficientNetB0": {
        "url": "https://drive.google.com/uc?id=1kfLjAHoGbYNBg12gU-vr9rCsnVX-kCFG",
        "filename": "efficientnet_finetuned.h5"
    },
    "MobileNetV3-Large": {
        "url": "https://drive.google.com/uc?id=1ARZryz9H5Bc832iRs2DiPyEjBL5BuNKR",
        "filename": "mobilenet_finetuned.h5"
    }
}

# ---------------------------------------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# ---------------------------------------------------------
def download_model_if_needed(model_name):
    config = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{config['filename']}"

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name}…"):
            gdown.download(config["url"], model_path, quiet=False)

    return model_path


# ---------------------------------------------------------
# SAFE MODEL LOADER (FINAL FIX)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):

    model_path = download_model_if_needed(model_name)

    # Try normal loading first (works for EfficientNet)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except Exception:
        st.warning(f"Normal load failed for {model_name}. Trying safe-loading…")

        # Special handling for MobileNetV3-Large
        try:
            base = tf.keras.applications.MobileNetV3Large(
                input_shape=(224, 224, 3),
                include_top=True,
                weights=None,
                classes=4
            )
            base.load_weights(model_path)
            return base

        except Exception as e:
            st.error(f"Failed to load MobileNetV3-Large: {e}")
            raise e


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#3B82F6;'>🧠 Brain MRI Tumor Detection</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["EfficientNetB0", "MobileNetV3-Large"],
    index=1
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1.3, 1])

with col1:
    if uploaded_file:
        arr, img_show = preprocess_image(uploaded_file)
        st.image(img_show, caption="Uploaded MRI Image", use_column_width=True)
    else:
        st.info("Upload an MRI scan to start.")

with col2:
    if uploaded_file:
        if st.button("Run Prediction", use_container_width=True):

            try:
                model = load_model(model_choice)

                with st.spinner("Analyzing MRI…"):
                    preds = model.predict(arr)[0]

                predicted_class = CLASS_NAMES[np.argmax(preds)]
                confidence = float(np.max(preds) * 100)

                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

                st.subheader("📊 Probability Distribution")
                table = {
                    "Tumor Type": CLASS_NAMES,
                    "Probability (%)": (preds * 100).round(2)
                }
                st.table(table)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
