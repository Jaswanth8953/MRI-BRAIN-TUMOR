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

# ---------------------------------------------------------
# CLASS NAMES & MODEL CONFIG
# ---------------------------------------------------------
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
# NAVIGATION SYSTEM
# ---------------------------------------------------------
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"  # "main" or "verification"

def show_navigation():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🏠 Back to Main", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    
    with col3:
        if st.button("🔍 Verify Models", use_container_width=True):
            st.session_state.current_page = "verification"
            st.rerun()
    
    return st.session_state.current_page

# ---------------------------------------------------------
# CUSTOM OBJECTS FOR MOBILENETV3
# ---------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

@tf.keras.utils.register_keras_serializable()
def hard_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

CUSTOM_OBJECTS = {
    'hard_swish': hard_swish,
    'HardSwish': hard_swish,
    'hard_sigmoid': hard_sigmoid,
    'HardSigmoid': hard_sigmoid,
    'relu6': tf.nn.relu6,
}

# ---------------------------------------------------------
# PREPROCESS IMPORTS FOR BOTH MODELS
# ---------------------------------------------------------
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess

# ---------------------------------------------------------
# MODEL LOADER (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    """Download (if needed) and load the selected model from Google Drive."""
    config = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{config['filename']}"
    
    # Download if not present
    if not os.path.exists(model_path):
        with st.spinner(f"📥 Downloading {model_name} from Google Drive..."):
            gdown.download(config["url"], model_path, quiet=False)
    
    # Try loading
    try:
        if model_name == "EfficientNetB0":
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        else:
            # MobileNetV3 with custom objects
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS
                )
                return model
            except Exception:
                # Fallback if saved without custom_objects
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    safe_mode=False
                )
                return model
    except Exception as e:
        st.error(f"Model load error for {model_name}: {str(e)[:150]}")
        return None

# ---------------------------------------------------------
# PREPROCESS IMAGE (MATCHES TRAINING PIPELINE)
# ---------------------------------------------------------
def preprocess_image(uploaded_file, model_choice):
    """
    Preprocess uploaded MRI according to the model's training preprocessing.
    EfficientNetB0  -> effnet_preprocess  ([-1, 1] scaling)
    MobileNetV3     -> mobilenet_preprocess (its own formula)
    """
    img = Image.open(uploaded_file).convert("RGB")
    display_img = img.copy()
    img = img.resize((224, 224))  # same as training
    arr = np.array(img).astype('float32')  # [0, 255]

    if model_choice == "EfficientNetB0":
        arr = effnet_preprocess(arr)
    else:
        arr = mobilenet_preprocess(arr)

    return np.expand_dims(arr, axis=0), display_img

# ---------------------------------------------------------
# VERIFICATION PAGE
# ---------------------------------------------------------
def verification_page():
    st.title("🔬 Model Verification Center")
    st.markdown("Verify that you're using **YOUR trained models** from Google Drive.")
    
    # Top navigation
    current_page = show_navigation()
    if current_page != "verification":
        return
    
    st.markdown("---")
    
    for model_name in MODEL_CONFIG.keys():
        st.header(f"🔍 Checking: {model_name}")
        
        col1, col2 = st.columns(2)
        config = MODEL_CONFIG[model_name]
        model_path = f"models/{config['filename']}"
        
        # ----- LEFT: FILE & HASH -----
        with col1:
            st.subheader("📁 YOUR Model File")
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024*1024)
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:16]
                
                st.success(f"✅ File exists: {file_size:.1f} MB")
                st.write(f"**Location:** `{model_path}`")
                st.write(f"**Source:** [Your Google Drive link]({config['url']})")
                st.write(f"**MD5 Hash (first 16 chars):** `{file_hash}...`")
                
                model = load_model(model_name)
                if model is not None:
                    st.success("✅ Loads as complete Keras model!")
                    test_input = np.random.random((1, 224, 224, 3)).astype('float32')
                    prediction = model.predict(test_input, verbose=0)
                    st.write(f"**Test Output Shape:** {prediction.shape}")
                    st.write(f"**Sum of probabilities:** {prediction.sum():.6f}")
                else:
                    st.error("❌ Could not load model. Check file or link.")
            else:
                st.warning("⚠️ Model file not downloaded yet.")
        
        # ----- RIGHT: COMPARE WITH STANDARD -----
        with col2:
            st.subheader("🔄 Comparison with Standard Model")
            
            standard = None
            try:
                if model_name == "EfficientNetB0":
                    standard = tf.keras.applications.EfficientNetB0(weights='imagenet')
                    st.write("**Standard EfficientNetB0 (ImageNet):**")
                else:
                    standard = tf.keras.applications.MobileNetV3Large(weights='imagenet')
                    st.write("**Standard MobileNetV3-Large (ImageNet):**")
                
                st.write(f"- Layers: {len(standard.layers)}")
                st.write(f"- Classes: 1000 (ImageNet)")
                st.write("- Weights: Pre-trained on ImageNet")
            except Exception as e:
                st.error(f"Cannot load standard reference model: {str(e)[:80]}")
            
            if os.path.exists(model_path):
                model = load_model(model_name)
                if model is not None:
                    st.write(f"\n**YOUR {model_name}:**")
                    st.write(f"- Layers: {len(model.layers)}")
                    try:
                        units = model.layers[-1].units
                        st.write(f"- Final layer units: {units}")
                        if units == 4:
                            st.success("✅ Looks like your 4-class tumor model.")
                        else:
                            st.warning("⚠️ Final layer is not 4 units. Check training code.")
                    except Exception:
                        st.info("ℹ️ Could not read final layer units.")
                else:
                    st.error("❌ Could not inspect model: load failed.")
        
        st.markdown("---")
    
    # Summary
    st.subheader("🎯 Verification Summary")
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.info("""
        **✅ EfficientNetB0**
        - Custom-trained with 4 tumor classes
        - Uses EfficientNet preprocessing
        - Recommended for predictions
        """)
    
    with col_sum2:
        st.info("""
        **ℹ️ MobileNetV3-Large**
        - Custom-trained with MobileNet preprocessing
        - Lighter and faster
        - Can be used experimentally
        """)
    
    st.markdown("---")
    if st.button("🏠 Go Back to MRI Analysis", use_container_width=True, type="primary"):
        st.session_state.current_page = "main"
        st.rerun()

# ---------------------------------------------------------
# MAIN PAGE (MRI ANALYSIS)
# ---------------------------------------------------------
def main_page():
    # Header row
    col_logo, col_title, col_nav = st.columns([1, 3, 1])
    
    with col_logo:
        st.markdown("### 🧠")
    
    with col_title:
        st.markdown(
            "<h1 style='text-align:center;'>Brain MRI Tumor Detection</h1>",
            unsafe_allow_html=True
        )
    
    with col_nav:
        if st.button("🔍 Verify Models", use_container_width=True):
            st.session_state.current_page = "verification"
            st.rerun()
    
    st.markdown("**Using YOUR trained EfficientNetB0 & MobileNetV3 models from Google Drive**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_choice = st.radio(
            "Select Model:",
            ["EfficientNetB0", "MobileNetV3-Large"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("🔍 Model Status")
        
        for model_name in ["EfficientNetB0", "MobileNetV3-Large"]:
            config = MODEL_CONFIG[model_name]
            model_path = f"models/{config['filename']}"
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024*1024)
                
                if model_name == "EfficientNetB0":
                    model = load_model(model_name)
                    if model is not None:
                        try:
                            if model.layers[-1].units == 4:
                                st.success(f"✅ {model_name}: 4-class custom model")
                            else:
                                st.warning(f"⚠️ {model_name}: Final layer = {model.layers[-1].units} units")
                        except Exception:
                            st.info(f"📦 {model_name}: {file_size:.1f} MB (couldn't read final layer)")
                    else:
                        st.error(f"❌ {model_name}: Load failed")
                else:
                    st.info(f"📦 {model_name}: {file_size:.1f} MB (custom MobileNetV3 model)")
            else:
                st.info(f"📥 {model_name}: Not downloaded yet")
        
        st.markdown("---")
        
        if model_choice == "EfficientNetB0":
            st.success("""
            **✅ RECOMMENDED**
            - Your EfficientNetB0 tumor classifier
            - 4 classes: glioma, meningioma, no_tumor, pituitary
            - Stable and verified
            """)
        else:
            st.warning("""
            **⚠️ EXPERIMENTAL**
            - MobileNetV3-Large custom model
            - Make sure preprocessing & training match
            """)
    
    # Main content: upload & prediction
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        # ----- LEFT: IMAGE & DETAILS -----
        with col1:
            arr, img_show = preprocess_image(uploaded_file, model_choice)
            st.image(img_show, caption="Uploaded MRI", use_column_width=True)
            
            with st.expander("🔧 Processing Details"):
                st.write(f"**Model:** {model_choice}")
                st.write(f"**Input Shape to model:** {arr.shape}")
                st.write(f"**Min value:** {arr.min():.3f}")
                st.write(f"**Max value:** {arr.max():.3f}")
                if model_choice == "EfficientNetB0":
                    st.write("**Preprocessing:** EfficientNetB0 `preprocess_input`")
                else:
                    st.write("**Preprocessing:** MobileNetV3 `preprocess_input`")
        
        # ----- RIGHT: ANALYSIS & RESULTS -----
        with col2:
            st.subheader("🔍 Analysis Results")
            
            if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    model = load_model(model_choice)
                    
                    if model:
                        with st.spinner("Analyzing MRI..."):
                            predictions = model.predict(arr, verbose=0)[0]
                        
                        predicted_idx = int(np.argmax(predictions))
                        predicted_class = CLASS_NAMES[predicted_idx]
                        confidence = float(predictions[predicted_idx] * 100)
                        
                        st.session_state.last_prediction = {
                            'predictions': predictions,
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'model_used': model_choice
                        }
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to load model! Check Verification page.")
            
            # Display last prediction if available
            if 'last_prediction' in st.session_state and st.session_state.last_prediction['model_used'] == model_choice:
                pred_data = st.session_state.last_prediction
                
                # Main prediction card
                st.markdown(f"""
                <div style='background:#eef6ff; padding:20px; border-radius:10px; border-left:6px solid #3b82f6; margin:20px 0;'>
                    <h3 style='margin-top:0; color:#1e40af;'>{pred_data['predicted_class'].replace('_', ' ').title()}</h3>
                    <h1 style='margin:10px 0; color:#3b82f6;'>{pred_data['confidence']:.1f}%</h1>
                    <p style='color:#64748b; margin:0;'>Confidence score</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence progress bar
                st.progress(
                    float(pred_data['confidence'] / 100.0),
                    text=f"Confidence Level: {pred_data['confidence']:.1f}%"
                )
                
                # All probabilities
                st.markdown("#### 📋 All Probabilities:")
                predictions = pred_data['predictions']
                predicted_idx = int(np.argmax(predictions))  # always defined
                
                for i in range(len(CLASS_NAMES)):
                    prob = float(predictions[i] * 100)
                    col_left, col_right = st.columns([3, 7])
                    
                    with col_left:
                        label = CLASS_NAMES[i].replace('_', ' ').title()
                        if i == predicted_idx:
                            st.markdown(f"**▶ {label}**")
                        else:
                            st.write(label)
                    
                    with col_right:
                        st.progress(float(predictions[i]), text=f"{prob:.1f}%")
                
                # Interpretation text
                st.markdown("---")
                st.markdown("#### 💡 Medical Interpretation (Not a medical diagnosis):")
                
                if pred_data['predicted_class'] == "no_tumor":
                    st.success("""
                    **✅ Model Suggestion: No Tumor**
                    - No strong tumor signature detected
                    - Regular medical check-ups are still recommended
                    """)
                elif pred_data['confidence'] > 80:
                    st.warning(f"""
                    **⚠️ Model Suggestion: High Confidence Tumor Detection**
                    - Strong indicators of **{pred_data['predicted_class'].replace('_', ' ')}**
                    - Urgent consultation with a radiologist is recommended
                    """)
                else:
                    st.info(f"""
                    **📊 Model Suggestion: Possible Tumor**
                    - Model leans towards **{pred_data['predicted_class'].replace('_', ' ')}**
                    - Clinical evaluation & expert opinion required
                    """)
            
            elif 'last_prediction' in st.session_state:
                st.info("👈 Run prediction with the selected model to update the results.")
            
            else:
                st.info(f"""
                ## 📊 Ready to Analyze
                
                Click **"Run Prediction"** to:
                1. Load the selected model
                2. Analyze the MRI scan
                3. View the tumor classification
                
                **Current Model:** {model_choice}
                """)
    
    else:
        # Welcome message when no file is uploaded
        st.info("""
        ## 👋 Welcome to Brain Tumor Classifier
        
        **How to use this app:**
        1. **Upload** an MRI brain scan (JPG/PNG)
        2. **Choose** a model in the sidebar (EfficientNetB0 recommended)
        3. **Click** "Run Prediction"
        4. **Review** the predicted tumor type and confidence
        
        **Recommended Model:** **EfficientNetB0**
        - ✅ Trained with EfficientNet preprocessing
        - ✅ 4 tumor classes (glioma, meningioma, no_tumor, pituitary)
        """)
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([2, 1])
    
    with footer_col1:
        st.caption("🧠 **Brain Tumor Classifier** | Using YOUR trained models")
        if uploaded_file and 'last_prediction' in st.session_state:
            st.caption(
                "Last prediction: "
                f"{st.session_state.last_prediction['predicted_class'].replace('_', ' ').title()}"
            )
    
    with footer_col2:
        st.caption(f"**Current Model:** {model_choice}")

# ---------------------------------------------------------
# MAIN APP ROUTER
# ---------------------------------------------------------
def main():
    if st.session_state.current_page == "verification":
        verification_page()
    else:
        main_page()

# ---------------------------------------------------------
# RUN THE APP
# ---------------------------------------------------------
if __name__ == "__main__":
    main()

# ---------------------------------------------------------
# CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Navigation buttons spacing */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"]
        > div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Better spacing for main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom expander styling */
    div[data-testid="stExpander"] details {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Button hover effect */
    .stButton > button:hover {
        transform: translateY(-2px);
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)
