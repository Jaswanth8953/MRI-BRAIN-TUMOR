import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import io
import base64

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
# UTILS
# ---------------------------------------------------------
def download_model_if_needed(model_name):
    config = MODEL_CONFIG[model_name]
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, config["filename"])

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} model..."):
            gdown.download(config["url"], model_path, quiet=False)

    return model_path


# ---------------------------------------------------------
# SAFE MODEL LOADER (FIXES MobileNetV3 ERROR)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    model_path = download_model_if_needed(model_name)

    # Try normal loading first
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except Exception as e:
        st.warning(f"Standard load failed for {model_name}. Applying safe loader…")

        try:
            tf.compat.v1.disable_eager_execution()

            with tf.compat.v1.get_default_graph().as_default():
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={"Functional": tf.keras.Model}
                )
            
            return model
        except Exception as e2:
            st.error(f"Model completely failed to load: {e2}")
            raise e2


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img


# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#4FC3F7;'>🧠 Brain Tumor Classification Dashboard</h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["EfficientNetB0", "MobileNetV3"]
)

st.sidebar.info("Upload MRI and click **Run Prediction**")

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
col1, col2 = st.columns([1.2, 1])

uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "jpeg", "png"])

with col1:
    if uploaded_file:
        img_array, img_disp = preprocess_image(uploaded_file)
        st.image(img_disp, caption="Uploaded MRI", use_column_width=True)

with col2:
    if uploaded_file:
        if st.button("Run Prediction", use_container_width=True):
            model = load_model(model_choice)

            with st.spinner("Analyzing MRI..."):
                preds = model.predict(img_array)[0]

            conf = float(np.max(preds) * 100)
            pred_class = CLASS_NAMES[np.argmax(preds)]

            # RESULTS
            st.success(f"Prediction: **{pred_class}**")
            st.info(f"Confidence: **{conf:.2f}%**")

            # PROBABILITY TABLE
            st.subheader("📊 Probability Distribution")

            prob_table = {
                "Tumor Type": CLASS_NAMES,
                "Probability (%)": (preds * 100).round(2)
            }

            st.table(prob_table)

