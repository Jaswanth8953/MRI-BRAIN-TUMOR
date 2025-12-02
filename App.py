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
# FIXED MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    config = MODEL_CONFIG[model_name]
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{config['filename']}"
    
    # Download if needed
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name}…"):
            gdown.download(config["url"], model_path, quiet=False)
    
    st.info(f"Loading {model_name} from: {model_path}")
    
    try:
        # ✅ CORRECT: Load the COMPLETE model (not just weights)
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                # Add any custom objects if needed
            }
        )
        st.success(f"{model_name} loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        
        # If MobileNetV3 fails, try to rebuild it correctly
        if "MobileNetV3" in model_name:
            st.warning("Attempting to rebuild MobileNetV3 architecture...")
            try:
                # Build the SAME architecture you used in training
                base_model = tf.keras.applications.MobileNetV3Large(
                    input_shape=(224, 224, 3),
                    include_top=False,  # ← CRITICAL: NO TOP!
                    weights='imagenet',
                    pooling='avg'
                )
                
                # Reconstruct the custom head (this should match your training)
                x = tf.keras.layers.Dense(512, activation='relu')(base_model.output)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
                
                model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
                
                # Load weights (only matching layers)
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                st.success("MobileNetV3 rebuilt with custom head!")
                return model
                
            except Exception as e2:
                st.error(f"Rebuild also failed: {e2}")
                raise e2
        else:
            raise e

# ---------------------------------------------------------
# CORRECT IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(uploaded_file, model_name):
    img = Image.open(uploaded_file).convert("RGB")
    
    # Save original for display
    display_img = img.copy()
    
    # Resize
    img = img.resize((224, 224))
    arr = np.array(img).astype('float32')
    
    # ✅ CORRECT PREPROCESSING FOR EACH MODEL:
    if "EfficientNet" in model_name:
        # EfficientNet expects pixels in [0, 255] range
        # and uses specific preprocessing
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif "MobileNetV3" in model_name:
        # MobileNetV3: normalize to [-1, 1]
        arr = arr / 127.5 - 1.0
    else:
        # Default normalization
        arr = arr / 255.0
    
    return np.expand_dims(arr, axis=0), display_img

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
        # Pass model_name to preprocessing
        arr, img_show = preprocess_image(uploaded_file, model_choice)
        st.image(img_show, caption="Uploaded MRI Image", use_column_width=True)
        
        # Show preprocessing info
        with st.expander("Preprocessing Details"):
            st.write(f"Model: {model_choice}")
            st.write(f"Input shape: {arr.shape}")
            st.write(f"Pixel range: [{arr.min():.2f}, {arr.max():.2f}]")
    else:
        st.info("Upload an MRI scan to start.")

with col2:
    if uploaded_file:
        if st.button("Run Prediction", use_container_width=True):
            try:
                model = load_model(model_choice)
                
                # Model summary (for debugging)
                with st.expander("Model Architecture"):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary[:20]))
                
                with st.spinner("Analyzing MRI…"):
                    preds = model.predict(arr, verbose=0)[0]
                
                predicted_class = CLASS_NAMES[np.argmax(preds)]
                confidence = float(np.max(preds) * 100)
                
                # Display results
                st.success(f"Prediction: **{predicted_class}**")
                
                # Confidence indicator
                if confidence > 80:
                    st.progress(confidence/100, text=f"Confidence: {confidence:.2f}%")
                elif confidence > 60:
                    st.progress(confidence/100, text=f"Confidence: {confidence:.2f}%")
                else:
                    st.warning(f"Low confidence: {confidence:.2f}%")
                
                # Probability distribution
                st.subheader("📊 Probability Distribution")
                
                # Create a bar chart
                chart_data = {
                    "Tumor Type": CLASS_NAMES,
                    "Probability": (preds * 100).round(2)
                }
                
                # Bar chart visualization
                st.bar_chart(
                    data={CLASS_NAMES[i]: preds[i] for i in range(4)},
                    use_container_width=True
                )
                
                # Table
                for i in range(4):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{CLASS_NAMES[i]}**")
                    with col_b:
                        st.write(f"{preds[i]*100:.2f}%")
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("""
                **Troubleshooting Tips:**
                1. Try a different MRI image
                2. Switch to EfficientNetB0 model
                3. Ensure image is clear and centered
                4. Try cropping to focus on brain region
                """)

# Add some test instructions
st.sidebar.markdown("---")
st.sidebar.info("""
**Testing Tips:**
1. Use clear, centered MRI images
2. Try both models for comparison
3. Check if predictions make sense
4. Upload test images with known labels
""")
