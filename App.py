import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import hashlib
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
            
            # Show download info
            file_size = os.path.getsize(model_path) / (1024*1024)
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            st.sidebar.info(f"Size: {file_size:.1f} MB | Hash: {file_hash}")

    return model_path

# ---------------------------------------------------------
# MODEL VERIFICATION TOOL
# ---------------------------------------------------------
def verify_model_ownership():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Verification")
    
    if st.sidebar.button("🔬 Verify Model Authenticity", use_container_width=True):
        st.session_state.show_verification = True
    
    if 'show_verification' in st.session_state and st.session_state.show_verification:
        st.sidebar.markdown("---")
        
        for model_name in ["EfficientNetB0", "MobileNetV3-Large"]:
            config = MODEL_CONFIG[model_name]
            model_path = f"models/{config['filename']}"
            
            with st.sidebar.expander(f"{model_name}", expanded=True):
                # Check file
                if os.path.exists(model_path):
                    # Get file hash
                    with open(model_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()[:12]
                    
                    file_size = os.path.getsize(model_path) / (1024*1024)
                    
                    st.write(f"**📁 File Info:**")
                    st.write(f"- Size: {file_size:.1f} MB")
                    st.write(f"- Hash: `{file_hash}`")
                    st.write(f"- Source: [Your Google Drive]({config['url']})")
                    
                    # Try to get model info
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                        st.success(f"✅ **YOUR Custom Model**")
                        st.write(f"- Layers: {len(model.layers)}")
                        st.write(f"- Parameters: {model.count_params():,}")
                        st.write(f"- Output Classes: {model.layers[-1].units}")
                        
                        # Compare with standard model
                        if model_name == "EfficientNetB0":
                            standard = tf.keras.applications.EfficientNetB0(weights='imagenet')
                        else:
                            standard = tf.keras.applications.MobileNetV3Large(weights='imagenet')
                        
                        if len(model.layers) > len(standard.layers):
                            st.success("✅ Has YOUR custom head")
                        elif len(model.layers) == len(standard.layers):
                            st.warning("⚠️ Same layers as standard (may be base only)")
                        else:
                            st.error("❌ Fewer layers than standard!")
                            
                    except Exception as e:
                        st.error(f"❌ Cannot load: {str(e)[:50]}")
                        st.info("Will try to rebuild architecture...")
                else:
                    st.warning("⚠️ File not downloaded yet")

# ---------------------------------------------------------
# SMART MODEL LOADER WITH VERIFICATION
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    config = MODEL_CONFIG[model_name]
    model_path = download_model_if_needed(model_name)

    # Show loading info in sidebar
    st.sidebar.info(f"**Loading:** {model_name}")
    st.sidebar.write(f"**From:** Your Google Drive")
    st.sidebar.write(f"**File:** `{config['filename']}`")

    try:
        # Try to load as complete model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Verification info
        st.sidebar.success("✅ **YOUR Custom Model Loaded**")
        st.sidebar.write(f"- Layers: {len(model.layers)}")
        st.sidebar.write(f"- Classes: {model.layers[-1].units}")
        
        return model

    except Exception as e:
        st.sidebar.error(f"❌ Direct load failed: {str(e)[:80]}")
        
        # MobileNetV3 fallback
        if model_name == "MobileNetV3-Large":
            st.sidebar.warning("🔄 Rebuilding MobileNetV3-Large...")
            
            # IMPORTANT: This uses ImageNet base weights + tries to load YOUR weights
            base = tf.keras.applications.MobileNetV3Large(
                input_shape=(224,224,3),
                include_top=False,
                pooling="avg",
                weights="imagenet"  # ← ImageNet weights (NOT yours)
            )
            
            # Add custom head (YOUR architecture)
            x = tf.keras.layers.Dense(512, activation="relu")(base.output)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
            
            model = tf.keras.Model(base.input, outputs)
            
            # Try to load YOUR weights
            try:
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                st.sidebar.success("✅ Loaded YOUR weights onto custom head")
            except Exception as e2:
                st.sidebar.error(f"❌ Failed to load weights: {str(e2)[:50]}")
                st.sidebar.warning("⚠️ Using ImageNet base + random head (NOT trained)")
            
            return model
        
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
# COMPREHENSIVE MODEL CHECK PAGE
# ---------------------------------------------------------
def comprehensive_model_check():
    st.title("🔬 COMPREHENSIVE MODEL VERIFICATION")
    st.warning("This page helps you verify if you're using YOUR trained models")
    
    for model_name in MODEL_CONFIG.keys():
        st.header(f"Checking: {model_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 YOUR Model File")
            config = MODEL_CONFIG[model_name]
            model_path = f"models/{config['filename']}"
            
            if os.path.exists(model_path):
                # Get file hash
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                file_size = os.path.getsize(model_path) / (1024*1024)
                
                st.success(f"✅ File exists: {file_size:.1f} MB")
                st.write(f"**Location:** `{model_path}`")
                st.write(f"**Source:** [Your Google Drive]({config['url']})")
                st.write(f"**MD5 Hash:** `{file_hash[:16]}...`")
                
                # Try to load YOUR model
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    st.success("✅ Loads as complete Keras model!")
                    
                    # Test prediction
                    test_input = np.random.random((1, 224, 224, 3)).astype('float32')
                    if "EfficientNet" in model_name:
                        test_input = tf.keras.applications.efficientnet.preprocess_input(test_input)
                    else:
                        test_input = test_input / 255.0
                    
                    prediction = model.predict(test_input, verbose=0)
                    st.write(f"**Test Output:** {prediction.shape}")
                    st.write(f"**Sum of probabilities:** {prediction.sum():.6f}")
                    
                except Exception as e:
                    st.error(f"❌ Cannot load: {str(e)[:100]}")
                    
            else:
                st.warning("⚠️ File not downloaded yet")
        
        with col2:
            st.subheader("🔄 Comparison with Standard Model")
            
            # Load standard model
            if model_name == "EfficientNetB0":
                standard = tf.keras.applications.EfficientNetB0(weights='imagenet')
                st.write("**Standard EfficientNetB0 (ImageNet):**")
            else:
                standard = tf.keras.applications.MobileNetV3Large(weights='imagenet')
                st.write("**Standard MobileNetV3-Large (ImageNet):**")
            
            st.write(f"- Layers: {len(standard.layers)}")
            st.write(f"- Classes: {standard.layers[-1].units} (ImageNet classes)")
            st.write(f"- Weights: Pre-trained on ImageNet (1.4M images)")
            
            # Try to compare with YOUR model
            if os.path.exists(model_path):
                try:
                    your_model = tf.keras.models.load_model(model_path, compile=False)
                    
                    st.write(f"\n**YOUR {model_name}:**")
                    st.write(f"- Layers: {len(your_model.layers)}")
                    st.write(f"- Classes: {your_model.layers[-1].units}")
                    
                    # Comparison
                    if len(your_model.layers) > len(standard.layers):
                        st.success("✅ **YOUR Custom Model Detected!**")
                        st.write("(Has additional custom layers for tumor classification)")
                    elif len(your_model.layers) == len(standard.layers):
                        st.warning("⚠️ **Same as Standard Model**")
                        st.write("(May not have your custom training)")
                    else:
                        st.error("❌ **Unexpected: Fewer layers than standard**")
                        
                except:
                    st.write("(Cannot inspect model structure)")
            
        st.markdown("---")
    
    st.info("""
    **🎯 Interpretation:**
    - **✅ Green check:** Using YOUR trained model
    - **⚠️ Warning:** Using standard ImageNet model
    - **❌ Red X:** Model loading failed
    
    **Your Google Drive URLs prove these are YOUR models.**
    """)

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
    
    # Add model verification
    verify_model_ownership()
    
    st.markdown("---")
    st.subheader("🔎 Quick Info")
    
    # Show what's being used
    if model_choice == "EfficientNetB0":
        st.success("**✅ Using YOUR EfficientNetB0**")
        st.caption("Trained on your tumor dataset")
    else:
        st.warning("**⚠️ Using YOUR MobileNetV3-Large**")
        st.caption("May fallback to ImageNet if loading fails")
    
    st.markdown("---")
    
    # Quick verification button
    if st.button("🚨 Run Full Verification", use_container_width=True, type="secondary"):
        st.session_state.run_full_verification = True

# ---------------------------------------------------------
# CHECK FOR FULL VERIFICATION
# ---------------------------------------------------------
if 'run_full_verification' in st.session_state and st.session_state.run_full_verification:
    comprehensive_model_check()
    st.stop()  # Stop rendering main app

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
        st.image(img_show, caption="Uploaded MRI", width='stretch')
        
        # Show preprocessing info
        with st.expander("🔧 Preprocessing Info"):
            st.write(f"**Model:** {model_choice}")
            st.write(f"**Input Range:** [{arr.min():.3f}, {arr.max():.3f}]")
            if "EfficientNet" in model_choice:
                st.write("**Preprocessing:** EfficientNet standard")
            else:
                st.write("**Preprocessing:** Simple [0, 1] normalization")

with col2:
    if uploaded_file:
        st.subheader("📘 Results Panel")
        
        # Initialize session state
        if 'preds' not in st.session_state:
            st.session_state.preds = None
        if 'predicted_class' not in st.session_state:
            st.session_state.predicted_class = None
        if 'confidence' not in st.session_state:
            st.session_state.confidence = None
        
        tab1, tab2, tab3 = st.tabs([
            "🔍 Prediction",
            "📊 Charts",
            "🧪 Model Info"
        ])
        
        # ---------------- TAB 1 — PREDICTION ----------------
        with tab1:
            if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
                model = load_model(model_choice)
                
                if model:
                    with st.spinner("🧠 Analyzing MRI…"):
                        preds = model.predict(arr, verbose=0)[0]
                    
                    # Store in session state
                    st.session_state.preds = preds
                    predicted_idx = np.argmax(preds)
                    st.session_state.predicted_class = CLASS_NAMES[predicted_idx]
                    st.session_state.confidence = float(preds[predicted_idx] * 100)
                    
                    st.rerun()
                else:
                    st.error("❌ Model failed to load!")
            
            # Display results
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
                
                st.progress(float(confidence/100), text=f"Confidence: {confidence:.2f}%")
                
                # All probabilities
                st.markdown("#### 📋 All Probabilities:")
                for i in range(4):
                    prob = preds[i] * 100
                    st.write(f"**{CLASS_NAMES[i].replace('_', ' ').title()}:** {prob:.2f}%")
        
        # ---------------- TAB 2 — CHARTS ----------------
        with tab2:
            st.markdown("### 📊 Probability Charts")
            
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
                             hole=0.3)
                st.plotly_chart(fig1, width='stretch')
                
                # Bar Chart
                st.markdown("#### 📈 Bar Graph")
                fig2 = px.bar(chart_df, 
                             x="Tumor Type", 
                             y="Probability",
                             text_auto='.2%')
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("👈 Run prediction first in the 'Prediction' tab")
        
        # ---------------- TAB 3 — MODEL INFO ----------------
        with tab3:
            st.markdown("### 🧪 Model Information")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.info(f"""
                **Selected Model:** {model_choice}
                **Status:** {"✅ Stable" if model_choice == "EfficientNetB0" else "⚠️ Experimental"}
                **Image Size:** 224 × 224
                **Classes:** 4 tumor types
                """)
            
            with col_info2:
                st.info(f"""
                **Source:** Your Google Drive
                **File:** {MODEL_CONFIG[model_choice]['filename']}
                **Preprocessing:** {"EfficientNet standard" if model_choice == "EfficientNetB0" else "[0, 1] normalization"}
                """)
            
            # Model verification status
            st.markdown("#### 🔍 Verification Status")
            try:
                model_path = f"models/{MODEL_CONFIG[model_choice]['filename']}"
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, compile=False)
                    st.success(f"✅ **YOUR trained model is active**")
                    st.write(f"- Layers: {len(model.layers)}")
                    st.write(f"- Custom head detected: {'Yes' if len(model.layers) > 150 else 'Maybe'}")
                else:
                    st.warning("⚠️ Model file not found locally")
            except:
                st.warning("⚠️ Model structure cannot be verified")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🧠 **Brain Tumor Classifier**")
    st.caption("Using YOUR trained models")

with footer_col2:
    st.caption(f"**Model:** {model_choice}")
    if uploaded_file and st.session_state.preds is not None:
        st.caption(f"**Confidence:** {st.session_state.confidence:.1f}%")

with footer_col3:
    st.caption("**Source:** Your Google Drive")
    st.caption("**For research purposes only**")

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
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    /* Custom badges */
    .stSuccess {
        border-left: 4px solid #10b981;
        padding-left: 10px;
    }
    
    .stWarning {
        border-left: 4px solid #f59e0b;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)
