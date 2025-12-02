import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
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
        with st.spinner(f"📥 Downloading {model_name}…"):
            gdown.download(config["url"], model_path, quiet=False)
            st.sidebar.success(f"✅ {model_name} downloaded!")

    return model_path

# ---------------------------------------------------------
# SMART MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    model_path = download_model_if_needed(model_name)

    try:
        # EfficientNet loads cleanly
        if model_name == "EfficientNetB0":
            return tf.keras.models.load_model(model_path, compile=False)

        # MobileNetV3 Large — safest loader
        elif model_name == "MobileNetV3-Large":
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={
                        "hard_swish": tf.keras.layers.Activation("hard_swish"),
                        "hard_sigmoid": tf.keras.layers.Activation("hard_sigmoid"),
                        "relu6": tf.nn.relu6
                    }
                )
                return model

            except:
                st.sidebar.warning("Reconstructing MobileNetV3-Large…")
                base = tf.keras.applications.MobileNetV3Large(
                    input_shape=(224,224,3),
                    include_top=False,
                    pooling="avg",
                    weights="imagenet"
                )

                x = tf.keras.layers.Dense(512, activation="relu")(base.output)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(256, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

                model = tf.keras.Model(base.input, outputs)

                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                return model

    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

# ---------------------------------------------------------
# PREPROCESS IMAGE
# ---------------------------------------------------------
def preprocess_image(uploaded_file, model_name):
    img = Image.open(uploaded_file).convert("RGB")
    disp = img.copy()
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32")

    if "EfficientNet" in model_name:
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif "MobileNet" in model_name:
        arr = arr / 255.0
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0), disp

# ---------------------------------------------------------
# HEADER DESIGN
# ---------------------------------------------------------
st.markdown("""
<div style='text-align:center; padding:20px; background:#1a2942; color:white; border-radius:10px;'>
    <h1>🧠 Brain MRI Tumor Detection</h1>
    <p>Deep Learning Powered Diagnosis Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    model_choice = st.selectbox(
        "Choose Model:",
        ["EfficientNetB0", "MobileNetV3-Large"],
        index=0
    )

    st.markdown("---")
    st.subheader("🔎 Info")
    if model_choice == "EfficientNetB0":
        st.success("EfficientNetB0 is highly stable and accurate.")
    else:
        st.warning("MobileNetV3 is experimental and may vary.")

    st.markdown("---")
    st.caption("Upload an MRI in the main panel →")

# ---------------------------------------------------------
# MAIN CONTENT WITH TABS
# ---------------------------------------------------------

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("📤 Upload MRI Image")

    uploaded_file = st.file_uploader(
        "Choose MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        arr, img_show = preprocess_image(uploaded_file, model_choice)
        st.image(img_show, caption="Uploaded MRI", width='stretch')  # FIXED: use_container_width → width

with col2:
    if uploaded_file:
        st.subheader("📘 Results Panel")
        
        # Initialize session state for predictions if not exists
        if 'preds' not in st.session_state:
            st.session_state.preds = None
        if 'predicted_class' not in st.session_state:
            st.session_state.predicted_class = None
        if 'confidence' not in st.session_state:
            st.session_state.confidence = None

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Prediction",
            "📊 Charts",
            "🧪 Model Info",
            "ℹ Help"
        ])

        # ---------------- TAB 1 — PREDICTION ----------------
        with tab1:
            if st.button("🚀 Run Prediction", use_container_width=True):
                model = load_model(model_choice)
                
                if model:
                    with st.spinner("Analyzing MRI…"):
                        preds = model.predict(arr, verbose=0)[0]
                    
                    # Store in session state for other tabs
                    st.session_state.preds = preds
                    predicted_idx = np.argmax(preds)
                    st.session_state.predicted_class = CLASS_NAMES[predicted_idx]
                    st.session_state.confidence = float(preds[predicted_idx] * 100)  # Convert to float
                    
                    # Force rerun to update all tabs
                    st.rerun()
                else:
                    st.error("❌ Model failed to load!")

            # Display results if we have predictions
            if st.session_state.preds is not None:
                preds = st.session_state.preds
                predicted_class = st.session_state.predicted_class
                confidence = st.session_state.confidence

                emoji = {
                    "glioma_tumor": "🟣",
                    "meningioma_tumor": "🔵",
                    "no_tumor": "🟢",
                    "pituitary_tumor": "🟡"
                }

                st.markdown(f"""
                    <div style='padding:15px; background:#eef6ff; border-radius:10px; border-left:6px solid #3b82f6;'>
                        <h3>{emoji[predicted_class]} {predicted_class.replace('_',' ').title()}</h3>
                        <h1 style='color:#3b82f6'>{confidence:.2f}%</h1>
                    </div>
                """, unsafe_allow_html=True)

                # FIXED: Convert confidence/100 to float for progress bar
                st.progress(float(confidence/100), text=f"Confidence: {confidence:.2f}%")

                # Show all probabilities
                st.markdown("#### 📋 All Class Probabilities:")
                for i in range(4):
                    prob = preds[i] * 100
                    st.write(f"**{CLASS_NAMES[i].replace('_', ' ').title()}:** {prob:.2f}%")

        # ---------------- TAB 2 — CHARTS ----------------
        with tab2:
            st.markdown("### 📊 Probability Charts")
            
            # Check if predictions exist
            if st.session_state.preds is not None:
                preds = st.session_state.preds
                
                chart_df = pd.DataFrame({
                    "Tumor Type": [c.replace('_', ' ').title() for c in CLASS_NAMES],
                    "Probability": preds
                })

                # Pie Chart
                st.markdown("#### 🥧 Pie Chart")
                fig1 = px.pie(chart_df, 
                             names="Tumor Type", 
                             values="Probability", 
                             hole=0.3,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig1, width='stretch')  # FIXED: use_container_width → width

                # Bar Chart
                st.markdown("#### 📈 Bar Graph")
                fig2 = px.bar(chart_df, 
                             x="Tumor Type", 
                             y="Probability",
                             color="Tumor Type",
                             text_auto='.2%')
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, width='stretch')  # FIXED: use_container_width → width
            else:
                st.info("👈 Run prediction first in the 'Prediction' tab")

        # ---------------- TAB 3 — MODEL INFO ----------------
        with tab3:
            st.markdown("### 🧪 Model Information")
            st.info(f"""
            **Selected Model:** {model_choice}  
            **Image Resolution:** 224 × 224  
            **Classes:** {', '.join(CLASS_NAMES)}
            **Status:** {"✅ Stable" if model_choice == "EfficientNetB0" else "⚠️ Experimental"}
            """)
            
            # Show preprocessing info
            st.markdown("#### 🔧 Preprocessing:")
            if "EfficientNet" in model_choice:
                st.write("EfficientNet standard preprocessing")
            else:
                st.write("Simple normalization [0, 1]")

        # ---------------- TAB 4 — HELP ----------------
        with tab4:
            st.markdown("### ℹ Help & Usage Guide")
            st.markdown("""
            **Steps to Use:**
            1. **Upload** MRI scan (JPG/PNG)  
            2. **Choose** model from sidebar  
            3. **Click** "Run Prediction" button  
            4. **Check** charts for deeper analysis  
            
            **Model Recommendations:**
            - **EfficientNetB0**: Best accuracy, most reliable
            - **MobileNetV3-Large**: Faster inference, experimental
            
            **Note:** This tool is for **research purposes only**.  
            Always consult with medical professionals for diagnosis.
            """)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🧠 **Brain Tumor Classifier**")

with footer_col2:
    st.caption(f"**Model:** {model_choice}")

with footer_col3:
    if uploaded_file and st.session_state.preds is not None:
        st.caption(f"**Confidence:** {st.session_state.confidence:.1f}%")
    else:
        st.caption("**Status:** Ready")

# ---------------------------------------------------------
# CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)
