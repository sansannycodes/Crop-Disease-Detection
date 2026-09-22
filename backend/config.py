"""
Backend Configuration module for Crop Disease Detection project.
Centralizes hyperparameter definitions, directory paths, and class labels.
"""

import os
from pathlib import Path

# Base Directory pointing to project root (parent of backend folder)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Model Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# Default Model Paths
MODEL_PATH = MODELS_DIR / "crop_disease_cnn.keras"
HISTORY_PLOT_PATH = OUTPUT_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

# 38 PlantVillage Crop Disease Classes
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

NUM_CLASSES = len(CLASS_NAMES)
