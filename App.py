import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import gdown
import os
from datetime import datetime

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

# ----------------- WHITE MEDICAL THEME -----------------
st.markdown("""
<style>
.stApp { background-color: #F4F7FA; }
.big-title { font-size: 2.2rem; font-weight: 700; color: #1F2937; }
.subtitle { font-size: 1rem; color: #4B5563; }
.result-card {
    padding: 1rem;
    border-radius: 10px;
    background: white;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.sidebar-title { font-weight: 600; color: #1F2937; }
</style>
""", unsafe_allow_html=True)

# ----------------- CONFIG -----------------
IMG_SIZE = (224, 224)

EFFICIENTNET_ID = "1kfLjAHoGbYNBg12gU-vr9rCsnVX-kCFG"
MOBILENET_ID    = "1ARZryz9H5Bc832iRs2DiPyEjBL5BuNKR"

MODEL_CONFIG = {
    "EfficientNetB0": {
        "url": f"https://drive.google.com/uc?id={EFFICIENTNET_ID}",
        "filename": "efficientnet_finetuned.h5"
    },
    "MobileNetV3": {
        "url": f"https://drive.google.com/uc?id={MOBILENET_ID}",
        "filename": "mobilenet_finetuned.h5"
    }
}

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------- HELPER: DOWNLOAD MODEL -----------------
def download_model_if_needed(model_name):
    cfg = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    path = f"models/{cfg['filename']}"
    if not os.path.exists(path):
        with st.spinner(f"Downloading {model_name} weights…"):
            gdown.download(cfg["url"], path, quiet=False)
    return path

# ----------------- SAFE MODEL LOADER (FIXED MOBILENET) -----------------
@st.cache_resource
def load_model(model_name):
    model_path = download_model_if_needed(model_name)

    # 1. Try normal load
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception:
        st.warning(f"Normal loading failed for {model_name}. Applying safe loader…")

    # 2. SAFE LOADER for MobileNetV3
    try:
        tf.compat.v1.disable_eager_execution()

        with tf.compat.v1.get_default_graph().as_default():
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"Functional": tf.keras.Model}
            )
        return model

    except Exception as e:
        st.error(f"Model loading failed completely for {model_name}: {e}")
        raise e

# ----------------- PREDICTION -----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0)

def predict(model_name, img):
    model = load_model(model_name)
    x = preprocess_image(img)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx], preds

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">Control Panel</p>', unsafe_allow_html=True)

    with st.expander("🕒 Session Tools", expanded=True):
        now = datetime.now()
        st.write("Date:", now.strftime("%Y-%m-%d"))
        st.write("Time:", now.strftime("%H:%M:%S"))
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.success("History cleared.")

    st.markdown("---")

    # DEFAULT: MobileNetV3 (your project model)
    model_choice = st.radio(
        "Select Model",
        ["EfficientNetB0", "MobileNetV3"],
        index=1
    )

    st.markdown("---")
    st.caption("Developed by Archana & Jaswanth")

# ----------------- HEADER -----------------
st.markdown('<p class="big-title">Brain MRI Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an MRI scan and classify tumor type using MobileNetV3 / EfficientNetB0.</p>', unsafe_allow_html=True)

# ----------------- TABS -----------------
tab_pred, tab_details, tab_history = st.tabs(["🔮 Prediction", "ℹ️ Model Details", "🕑 History"])

# ----------------- TAB 1: PREDICTION -----------------
with tab_pred:
    left, right = st.columns([1.3, 1])

    with left:
        uploaded_file = st.file_uploader("Upload MRI (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded MRI", use_column_width=True)
        else:
            st.info("Please upload an MRI image.")

    with right:
        st.markdown("### Model Output")

        if uploaded_file:
            if st.button("Run Prediction", use_container_width=True):
                try:
                    label, conf, probs = predict(model_choice, img)

                    # Save history
                    st.session_state["history"].append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_choice,
                        "prediction": label,
                        "confidence": round(conf * 100, 2)
                    })

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"**Model:** {model_choice}")
                    st.markdown(f"**Prediction:** `{label}`")
                    st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Probability Table
                    df = pd.DataFrame({"Tumor Type": CLASS_NAMES, "Probability": probs})
                    df["Probability (%)"] = (df["Probability"] * 100).round(2)
                    df = df.set_index("Tumor Type")

                    st.markdown("#### Probability Distribution")
                    st.dataframe(df[["Probability (%)"]])
                    st.bar_chart(df["Probability"])

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

        else:
            st.info("Upload an MRI to generate prediction.")

# ----------------- TAB 2: MODEL DETAILS -----------------
with tab_details:
    st.write("### Model Details")
    st.write(f"Selected Model: **{model_choice}**")
    st.write("Classes:", ", ".join(CLASS_NAMES))

# ----------------- TAB 3: HISTORY -----------------
with tab_history:
    st.write("### Prediction History")
    if len(st.session_state["history"]) == 0:
        st.info("No predictions yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["history"]))
