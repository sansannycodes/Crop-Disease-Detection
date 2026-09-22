"""
Backend Dataset loading and preprocessing module.
Handles OpenCV/PIL image loading, normalization, and Keras data generators.
"""

import os
from pathlib import Path
import numpy as np

# Try importing OpenCV, fallback to PIL if OpenCV is not yet installed
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    from PIL import Image

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    HAS_TF = True
except ImportError:
    HAS_TF = False

from backend import config


def load_image_cv2(image_path: str, target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """
    Reads an image from disk, converts to RGB, and resizes it.
    Uses OpenCV if available, otherwise falls back to PIL.
    """
    image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if HAS_OPENCV:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"OpenCV failed to decode image at path: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))
        return img_resized
    else:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize((target_size[1], target_size[0]))
            return np.array(img_resized)


def preprocess_single_image(image_path: str, target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """
    Preprocesses a single image for model inference: reads, resizes, normalizes [0,1],
    and expands dimensions to (1, height, width, 3).
    """
    img_rgb = load_image_cv2(image_path, target_size)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch


def get_data_generators(data_dir: Path = config.DATASET_DIR,
                         batch_size: int = config.BATCH_SIZE,
                         img_size: tuple = config.IMG_SIZE):
    """
    Creates train and validation data generators with data augmentation for training.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required for data generator creation. Please run 'pip install -r requirements.txt'")

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders 'train' and 'valid' were not found inside {data_dir}. "
            f"Please follow instructions in data/README.md to setup the dataset."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = valid_datagen.flow_from_directory(
        directory=str(valid_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator
