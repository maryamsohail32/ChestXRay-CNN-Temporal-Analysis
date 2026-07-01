import os
import subprocess
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Chest X-Ray Analysis", layout="wide")
st.title("🫁 Chest X-Ray CNN Temporal Analysis")

MODEL_PATH = "best_model.h5"

# If the model file doesn't exist, trigger training automatically
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Trained model weights file (`best_model.h5`) not found.")
    
    with st.spinner("🔄 Initializing training run directly via 'cnn_xray_pneumonia.py' to generate the model... This will take a while."):
        try:
            # Executes your training script in the background
            result = subprocess.run(["python", "cnn_xray_pneumonia.py"], capture_output=True, text=True)
            
            if os.path.exists(MODEL_PATH):
                st.success("✅ Training completed and 'best_model.h5' generated successfully!")
                st.rerun()
            else:
                st.error("❌ Training script ran but failed to output 'best_model.h5'. Check your script log output below.")
                st.text(result.stdout)
                st.text(result.stderr)
        except Exception as e:
            st.error(f"❌ Failed to execute training script: {e}")

# If model exists (or was just created), load and run prediction interface
if os.path.exists(MODEL_PATH):
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_my_model()
    st.success("✅ CNN Model Loaded and Ready!")

    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-Ray', width=300)
        
        st.write("Preprocessing image data...")
        img_resized = image.resize((224, 224)) 
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("Run CNN Diagnosis"):
            prediction = model.predict(img_array)
            st.write(f"Analysis complete. Raw Prediction Score: {prediction[0][0]:.4f}")
