"""
Frontend Interactive Streamlit Web Application for Crop Disease Detection.
Provides drag-and-drop image upload, image preprocessing preview,
and CNN model disease classification with actionable crop health recommendations.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.dataset import preprocess_single_image

# Conditional TensorFlow import
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Page Configuration
st.set_page_config(
    page_title="Crop Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E4620;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4A6B4C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F0F7F1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model(model_path: str):
    """Loads and caches the trained Keras CNN model."""
    if not HAS_TF or not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def format_class_name(raw_name: str) -> str:
    """Formats raw class string (e.g. Tomato___Early_blight) into clean readable title."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").replace(",", "")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return f"{crop} - {disease}"


def main():
    st.markdown('<div class="main-title">🌿 Crop Disease Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated AI Plant Leaf Health Diagnosis using Convolutional Neural Networks</div>', unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("📋 Project Info")
        st.write("**Model Architecture:** CNN (Custom Sequential)")
        st.write("**Target Crop Classes:** 38 Disease Categories")
        st.write("**Input Resolution:** 224 x 224 pixels")
        st.write("**Supported Formats:** JPG, JPEG, PNG")
        st.markdown("---")
        st.subheader("💡 Usage Instructions")
        st.markdown("""
        1. Upload a clear photograph of a crop leaf.
        2. View the preprocessed input preview.
        3. Review AI disease predictions and confidence scores.
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Upload Leaf Image")
        uploaded_file = st.file_uploader("Choose a leaf image file...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Save temporary file for prediction processing
            temp_path = config.OUTPUT_DIR / "temp_upload.jpg"
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            image.convert("RGB").save(temp_path)

    with col2:
        st.subheader("🔍 AI Diagnosis & Results")

        if uploaded_file is not None:
            model = load_trained_model(str(config.MODEL_PATH))

            if model is None:
                st.warning(
                    f"⚠️ Trained model binary not found at `{config.MODEL_PATH}`.\n\n"
                    "Please run `python backend/train.py` to train the CNN model or download pre-trained weights."
                )
            else:
                with st.spinner("Analyzing leaf image using CNN..."):
                    input_tensor = preprocess_single_image(str(temp_path), target_size=config.IMG_SIZE)
                    predictions = model.predict(input_tensor, verbose=0)[0]

                    top_indices = np.argsort(predictions)[::-1][:3]
                    top_class = config.CLASS_NAMES[top_indices[0]]
                    top_conf = float(predictions[top_indices[0]]) * 100.0

                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h3>Primary Diagnosis</h3>
                            <h2>{format_class_name(top_class)}</h2>
                            <p>Confidence: <b>{top_conf:.2f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("---")
                    st.write("### Top Probable Predictions")
                    for idx in top_indices:
                        raw_cls = config.CLASS_NAMES[idx]
                        conf_val = float(predictions[idx])
                        st.write(f"**{format_class_name(raw_cls)}**")
                        st.progress(conf_val)
                        st.write(f"Confidence: `{conf_val * 100:.2f}%`")
        else:
            st.info("👈 Please upload a crop leaf image to initiate disease classification.")


if __name__ == "__main__":
    main()
