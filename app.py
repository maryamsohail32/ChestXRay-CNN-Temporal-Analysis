import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Chest X-Ray Analysis", layout="wide")
st.title("🫁 Chest X-Ray CNN Temporal Analysis")

# Check if model exists, if not, give a clear message
MODEL_PATH = "best_model.h5" 

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Trained model file (`best_model.h5`) not found in the repository.")
    st.info("Please run your training script (`cnn_xray_pneumonia.py`) locally to generate the model file, then upload it here.")
else:
    # Load the model safely
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_my_model()
    st.success("✅ CNN Model Loaded Successfully!")

    # Image Uploader Component
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-Ray', width=300)
        
        # Simple preprocessing placeholder matching standard CNN inputs
        st.write("Analyzing visual features...")
        img_resized = image.resize((224, 224)) 
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_gradient_selection = np.expand_dims(img_array, axis=0)
        
        # Predict button
        if st.button("Run CNN Diagnosis"):
            prediction = model.predict(img_array)
            # Customize this logic based on your specific model output classes
            st.write(f"Analysis complete. Raw Output Score: {prediction[0][0]:.4f}")
