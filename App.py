import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import gdown
import os
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

# ---------- LIGHT HOSPITAL THEME ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1F2937;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 1rem;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CONFIG ----------
IMG_SIZE = (224, 224)

# Your Google Drive IDs (already correct)
EFFICIENTNET_ID = "1kfLjAHoGbYNBg12gU-vr9rCsnVX-kCFG"
MOBILENET_ID   = "1ARZryz9H5Bc832iRs2DiPyEjBL5BuNKR"

MODEL_CONFIG = {
    "EfficientNetB0": {
        "url": f"https://drive.google.com/uc?id={EFFICIENTNET_ID}",
        "filename": "efficientnet_finetuned.h5",
        "description": "High-capacity CNN, very strong overall accuracy.",
    },
    "MobileNetV3": {
        "url": f"https://drive.google.com/uc?id={MOBILENET_ID}",
        "filename": "mobilenet_finetuned.h5",
        "description": "Lightweight CNN, fast inference. Main model for this project.",
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
    st.session_state["history"] = []  # list of dicts


# ---------- MODEL HELPERS ----------
def download_model_if_needed(model_name: str) -> str:
    cfg = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", cfg["filename"])

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} weights from Google Drive..."):
            gdown.download(cfg["url"], model_path, quiet=False)

    return model_path


@st.cache_resource
def load_model(model_name: str):
    """Load model from local path (download from Drive if needed)."""
    model_path = download_model_if_needed(model_name)
    # compile=False to avoid needing original training config
    model = tf.keras.models.load_model(model_path, compile=False)
    # optional compile for metrics usage
    try:
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception:
        # some models may not need compile for inference
        pass
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Same style as training: RGB -> resize -> /255."""
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model_name: str, image: Image.Image):
    """Run inference with selected model."""
    model = load_model(model_name)
    x = preprocess_image(image)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    return label, confidence, preds


# ---------- SIDEBAR (TOOLBAR) ----------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Control Panel</div>', unsafe_allow_html=True)

    with st.expander("🕒 Session tools", expanded=True):
        now = datetime.now()
        st.write("**Date:**", now.strftime("%Y-%m-%d"))
        st.write("**Time:**", now.strftime("%H:%M:%S"))
        if st.button("🧹 Clear prediction history"):
            st.session_state["history"] = []
            st.success("History cleared.")

    st.markdown("---")

    # Default to MobileNetV3 (your main project model)
    model_choice = st.radio(
        "Model selection",
        options=list(MODEL_CONFIG.keys()),
        index=1,  # 0 = EfficientNet, 1 = MobileNet
        help="MobileNetV3 is the main model used in your project.",
    )

    st.markdown("**Model description**")
    st.write(MODEL_CONFIG[model_choice]["description"])

    st.markdown("---")
    st.caption("Developed by Archana & Jaswanth – Brain MRI Tumor Detection")


# ---------- HEADER ----------
st.markdown('<div class="big-title">Brain MRI Tumor Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a brain MRI image and obtain tumor predictions using your '
    'trained MobileNetV3 / EfficientNetB0 models.</div>',
    unsafe_allow_html=True,
)

# ---------- TABS ----------
tab_pred, tab_details, tab_history = st.tabs(
    ["🔮 Prediction", "ℹ️ Model Details", "🕑 Prediction History"]
)

# ===== TAB 1: PREDICTION =====
with tab_pred:
    col_left, col_right = st.columns([1.2, 1])

    # LEFT: Image upload
    with col_left:
        uploaded_file = st.file_uploader(
            "Upload MRI image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
        else:
            st.info("Please upload a brain MRI image to start.")

    # RIGHT: Prediction card
    with col_right:
        st.markdown("#### Model Output")

        if uploaded_file is not None:
            if st.button("Run prediction", use_container_width=True):
                try:
                    with st.spinner("Running inference..."):
                        label, conf, probs = predict(model_choice, image)

                    # Save to history
                    st.session_state["history"].append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": model_choice,
                            "prediction": label,
                            "confidence": round(conf * 100, 2),
                        }
                    )

                    # Result card
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"**Model:** {model_choice}")
                    st.markdown(f"**Predicted class:** `{label}`")
                    st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Probability table + bar chart
                    prob_df = pd.DataFrame(
                        {"Tumor Type": CLASS_NAMES, "Probability": probs}
                    )
                    prob_df["Probability (%)"] = (
                        prob_df["Probability"] * 100
                    ).round(2)
                    prob_df = prob_df.set_index("Tumor Type")

                    st.markdown("##### Probability Distribution")
                    st.dataframe(prob_df[["Probability (%)"]])
                    st.bar_chart(prob_df["Probability"])

                except Exception as e:
                    st.error(
                        f"Prediction failed for **{model_choice}**. "
                        f"Technical error: {e}"
                    )
        else:
            st.info("Prediction will appear here after you upload an image and click the button.")

# ===== TAB 2: MODEL DETAILS =====
with tab_details:
    st.markdown("### Model Details")
    st.write(f"**Selected model:** `{model_choice}`")
    st.write(MODEL_CONFIG[model_choice]["description"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input size**")
        st.write(f"{IMG_SIZE[0]} × {IMG_SIZE[1]} RGB")
        st.markdown("**Number of classes**")
        st.write(len(CLASS_NAMES))
        st.markdown("**Classes**")
        st.write(", ".join(CLASS_NAMES))

    with col2:
        st.markdown("**Prediction pipeline**")
        st.markdown(
            """
            1. Read uploaded MRI image.  
            2. Convert to RGB, resize to 224×224, normalize to [0, 1].  
            3. Forward pass through the selected CNN (MobileNetV3 / EfficientNetB0).  
            4. Apply softmax over 4 tumor classes.  
            5. Display highest-probability class and full distribution.  
            """
        )

# ===== TAB 3: HISTORY =====
with tab_history:
    st.markdown("### Prediction History (current session)")
    if len(st.session_state["history"]) == 0:
        st.info("No predictions yet. Run a prediction in the first tab.")
    else:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df)
