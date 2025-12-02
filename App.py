# ---------------------------------------------------------
# MAIN CONTENT WITH TABS
# ---------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        arr, img_show = preprocess_image(uploaded_file, model_choice)
        st.image(img_show, caption="Uploaded MRI Scan", use_container_width=True)

        with st.expander("🔧 Preprocessing Details", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Model", model_choice)
                st.metric("Input Shape", str(arr.shape))
            with col_b:
                st.metric("Pixel Range", f"[{arr.min():.3f}, {arr.max():.3f}]")
                st.metric("Normalization", "[0,1]" if "MobileNetV3" in model_choice else "EfficientNet")

with col2:
    if uploaded_file:
        st.subheader("📘 Results Panel")

        # -----------------------
        # NEW: TABS FOR CLEAN UI
        # -----------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Prediction",
            "📊 Charts",
            "🧪 Model Info",
            "ℹ Help & Notes"
        ])

        with tab1:
            st.markdown("### 🔍 Prediction Results")
            if st.button("🚀 Run Prediction", type="primary", use_container_width=True):

                model = load_model(model_choice)

                if model is None:
                    st.error("❌ Model failed to load.")
                else:
                    with st.spinner("🧠 Analyzing MRI Scan..."):
                        preds = model.predict(arr, verbose=0)[0]

                    predicted_idx = np.argmax(preds)
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = float(np.max(preds) * 100)

                    emoji_map = {
                        "glioma_tumor": "🟣",
                        "meningioma_tumor": "🔵",
                        "no_tumor": "🟢",
                        "pituitary_tumor": "🟡"
                    }

                    # Clean result card
                    st.markdown(f"""
                    <div style="
                        background:white;
                        padding:20px;
                        border-radius:12px;
                        border-left:8px solid #3B82F6;
                        box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                        <h3 style="margin:0;">{emoji_map[predicted_class]} {predicted_class.replace('_',' ').title()}</h3>
                        <h1 style="color:#3B82F6; margin-top:5px;">{confidence:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(confidence/100, text=f"Confidence Score: {confidence:.1f}%")

                    # Detailed Probabilities
                    st.markdown("### 📊 Detailed Probabilities")
                    for i in range(4):
                        prob = preds[i] * 100
                        name = CLASS_NAMES[i].replace("_"," ").title()

                        colA, colB = st.columns([3,7])
                        with colA:
                            if i == predicted_idx:
                                st.markdown(f"**▶ {name}**")
                            else:
                                st.write(name)
                        with colB:
                            st.progress(float(preds[i]), text=f"{prob:.2f}%")

        # -----------------------------------
        # TAB 2: BEAUTIFUL PIE CHART & BAR
        # -----------------------------------
        with tab2:
            st.markdown("### 📊 Advanced Visualizations")

            if uploaded_file:
                import pandas as pd

                chart_df = pd.DataFrame({
                    "Tumor Type": CLASS_NAMES,
                    "Probability": preds
                })

                # Pie chart
                st.markdown("#### 🥧 Prediction Pie Chart")
                st.plotly_chart({
                    "data": [{
                        "labels": chart_df["Tumor Type"],
                        "values": chart_df["Probability"],
                        "type": "pie",
                        "hole": .3
                    }],
                    "layout": {"height": 400}
                })

                st.markdown("#### 📈 Bar Chart")
                st.bar_chart(chart_df.set_index("Tumor Type"))

        # -----------------------------------
        # TAB 3: Model Info
        # -----------------------------------
        with tab3:
            st.markdown("### 🧪 Model Information")

            if model_choice == "EfficientNetB0":
                st.success("""
                **EfficientNetB0**
                - High accuracy  
                - Stable & recommended  
                - Uses EfficientNet preprocessing  
                """)
            else:
                st.warning("""
                **MobileNetV3-Large**
                - Fast model  
                - Experimental  
                - Uses simple normalization  
                """)

            st.info("""
            **Image Size:** 224×224  
            **Channels:** RGB  
            **Framework:** TensorFlow 2.x  
            """)

        # -----------------------------------
        # TAB 4: Help Menu
        # -----------------------------------
        with tab4:
            st.markdown("### ℹ How to Use This Tool")

            st.markdown("""
            1. Upload an MRI brain scan  
            2. Choose model  
            3. Click **Run Prediction**  
            4. View results, charts & interpretation  
            5. Compare multiple models  

            **Note:** This tool is for academic use only.
            """)
