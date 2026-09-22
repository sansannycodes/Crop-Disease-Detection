"""
Streamlit web interface for Crop Disease Detection.
Provides file upload for leaf images, image preprocessing, and model prediction display.
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
    page_title="Crop Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1E4620;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4A6B4C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F0F7F1;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #2E7D32;
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
    """Formats raw class folder names into readable titles."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").replace(",", "")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return f"{crop} - {disease}"


def main():
    st.markdown('<div class="main-title">Crop Disease Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Plant Leaf Classification using Convolutional Neural Networks</div>', unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("Project Details")
        st.write("Model Type: CNN (Custom Sequential)")
        st.write("Number of Classes: 38 Categories")
        st.write("Input Size: 224 x 224 pixels")
        st.write("Supported Formats: JPG, JPEG, PNG")
        st.markdown("---")
        st.subheader("Instructions")
        st.markdown("""
        1. Upload a leaf image.
        2. View the uploaded image preview.
        3. Review the predicted disease classification.
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Leaf Image")
        uploaded_file = st.file_uploader("Select an image file...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            temp_path = config.OUTPUT_DIR / "temp_upload.jpg"
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            image.convert("RGB").save(temp_path)

    with col2:
        st.subheader("Prediction Results")

        if uploaded_file is not None:
            model = load_trained_model(str(config.MODEL_PATH))

            if model is None:
                st.warning(
                    f"Trained model file not found at `{config.MODEL_PATH}`.\n\n"
                    "Run `python backend/train.py` to train the model or download pre-trained weights."
                )
            else:
                with st.spinner("Analyzing image..."):
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
                    st.write("### Top Predictions")
                    for idx in top_indices:
                        raw_cls = config.CLASS_NAMES[idx]
                        conf_val = float(predictions[idx])
                        st.write(f"**{format_class_name(raw_cls)}**")
                        st.progress(conf_val)
                        st.write(f"Confidence: `{conf_val * 100:.2f}%`")
        else:
            st.info("Upload a leaf image to see predictions.")


if __name__ == "__main__":
    main()
