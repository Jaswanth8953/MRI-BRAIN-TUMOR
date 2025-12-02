import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import gdown
import os
from datetime import datetime

# ---------- BASIC STYLING ----------
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

# Optional custom CSS for nicer medical UI
st.markdown(
    """
    <style>
    .main {
        background-color: #0B1020;
        color: #F5F5F5;
    }
    .stApp {
        background-color: #0B1020;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #D0D0D0;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #111829;
        border: 1px solid #2F3A5C;
        color: #E0E0E0;
    }
    .sidebar .sidebar-content {
        background-color: #111829 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CONFIG ----------
IMG_SIZE = (224, 224)

# Google Drive File IDs (YOURS ARE ALREADY SET)
EFFICIENTNET_ID = "1kfLjAHoGbYNBg12gU-vr9rCsnVX-kCFG"
MOBILENET_ID   = "1ARZryz9H5Bc832iRs2DiPyEjBL5BuNKR"

MODEL_CONFIG = {
    "EfficientNetB0": {
        "url": f"https://drive.google.com/uc?id={EFFICIENTNET_ID}",
        "filename": "efficientnet_finetuned.h5",
        "description": "Accurate & robust model (EfficientNetB0).",
    },
    "MobileNetV3": {
        "url": f"https://drive.google.com/uc?id={MOBILENET_ID}",
        "filename": "mobilenet_finetuned.h5",
        "description": "Fast lightweight model (MobileNetV3).",
    },
}

CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------- FUNCTIONS ----------
def download_model_if_needed(model_name: str) -> str:
    cfg = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", cfg["filename"])

    if not os.path.exists(model_path):
        with st.spinner(f"⬇️ Downloading {model_name} weights..."):
            gdown.download(cfg["url"], model_path, quiet=False)

    return model_path


@st.cache_resource
def load_model(model_name: str):
    path = download_model_if_needed(model_name)
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def predict(model_name: str, img: Image.Image):
    model = load_model(model_name)
    x = preprocess_image(img)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx], preds


# ---------- SIDEBAR TOOLBAR ----------
with st.sidebar:
    st.markdown("### 🧭 Control Panel")

    # Date & time expander like toolbar
    with st.expander("🕒 Session Tools", expanded=True):
        now = datetime.now()
        st.write("**Date:**", now.strftime("%Y-%m-%d"))
        st.write("**Time:**", now.strftime("%H:%M:%S"))

        if st.button("🧹 Clear History"):
            st.session_state["history"] = []
            st.success("History cleared!")

    st.markdown("---")

    # Model selector
    model_choice = st.radio(
        "Choose model",
        list(MODEL_CONFIG.keys()),
        index=0,
    )

    st.markdown("#### Model Info")
    st.write(MODEL_CONFIG[model_choice]["description"])

    st.markdown("---")
    st.markdown("Made with ❤️ by **Archana & Jaswanth**")


# ---------- HEADER ----------
st.markdown('<div class="big-title">Brain MRI Tumor Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered medical image analysis using EfficientNetB0 & MobileNetV3.</div>',
    unsafe_allow_html=True,
)

st.write("")

# ---------- MAIN TABS ----------
tab_pred, tab_details, tab_training, tab_explain, tab_history = st.tabs(
    ["🔮 Prediction", "ℹ️ Model Details", "📈 Training Metrics", "🔥 Explainability", "🕑 History"]
)

# ==================== TAB 1: PREDICTION ======================
with tab_pred:
    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        uploaded = st.file_uploader("Upload Brain MRI (JPG/PNG)", ["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Upload an MRI image to begin.")

    with col_right:
        if uploaded and st.button("🔍 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analyzing MRI..."):
                label, conf, probs = predict(model_choice, img)

            # Save history
            st.session_state["history"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_choice,
                "prediction": label,
                "confidence": round(conf * 100, 2),
            })

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"### 🧠 Predicted: **{label}**")
            st.markdown(f"#### Confidence: **{conf*100:.2f}%**")
            st.markdown(f"**Model:** {model_choice}")
            st.markdown("</div>", unsafe_allow_html=True)

            prob_df = pd.DataFrame({"Tumor Type": CLASS_NAMES, "Probability": probs})
            prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
            prob_df = prob_df.set_index("Tumor Type")

            st.markdown("#### Probability Distribution")
            st.dataframe(prob_df[["Probability (%)"]])
            st.bar_chart(prob_df["Probability"])
        else:
            st.info("Prediction will appear here after running the model.")

# ==================== TAB 2: MODEL DETAILS ======================
with tab_details:
    st.markdown("### ℹ️ Model Details")
    st.write(f"**Selected model:** `{model_choice}`")
    st.write(MODEL_CONFIG[model_choice]["description"])

    st.markdown("#### Class Labels")
    st.write(", ".join(CLASS_NAMES))

    st.markdown("#### Input Info")
    st.write(f"Input size: {IMG_SIZE[0]} × {IMG_SIZE[1]} RGB")

# ==================== TAB 3: TRAINING METRICS ======================
with tab_training:
    st.markdown("### 📈 Training & Metrics (Upload images in assets/ folder)")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### EfficientNetB0")
        try:
            st.image("assets/efficientnet_training_curves.png")
        except:
            st.info("Upload assets/efficientnet_training_curves.png")

        try:
            st.image("assets/efficientnet_confusion_matrix.png")
        except:
            st.info("Upload assets/efficientnet_confusion_matrix.png")

    with colB:
        st.markdown("#### MobileNetV3")
        try:
            st.image("assets/mobilenet_training_curves.png")
        except:
            st.info("Upload assets/mobilenet_training_curves.png")

        try:
            st.image("assets/mobilenet_confusion_matrix.png")
        except:
            st.info("Upload assets/mobilenet_confusion_matrix.png")

# ==================== TAB 4: EXPLAINABILITY ======================
with tab_explain:
    st.markdown("### 🔥 Explainability (Grad-CAM)")

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image("assets/gradcam/gradcam_example_1.png")
        except:
            st.info("Upload assets/gradcam/gradcam_example_1.png")

    with col2:
        try:
            st.image("assets/gradcam/gradcam_example_2.png")
        except:
            st.info("Upload assets/gradcam/gradcam_example_2.png")

# ==================== TAB 5: HISTORY ======================
with tab_history:
    st.markdown("### 🕑 Prediction History")
    if len(st.session_state["history"]) == 0:
        st.info("No predictions yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["history"]))
