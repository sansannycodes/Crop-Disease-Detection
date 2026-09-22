"""
Unit tests for Crop Disease Detection project pipeline.
Validates config parameters, model architecture construction, and preprocessing logic.
Compatible with standard python unittest and pytest.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.dataset import preprocess_single_image, load_image_cv2

# Conditional TensorFlow import for testing environment compatibility
try:
    import tensorflow as tf
    from backend.model import build_cnn_model
    HAS_TF = True
except ImportError:
    HAS_TF = False


class TestCropDiseasePipeline(unittest.TestCase):

    def test_config_parameters(self):
        """Verify config contains expected parameters and 38 classes."""
        self.assertEqual(config.NUM_CLASSES, 38)
        self.assertEqual(len(config.CLASS_NAMES), 38)
        self.assertEqual(config.IMG_HEIGHT, 224)
        self.assertEqual(config.IMG_WIDTH, 224)
        self.assertEqual(config.INPUT_SHAPE, (224, 224, 3))

    def test_cnn_model_architecture(self):
        """Verify CNN model builds with expected input/output shapes if TensorFlow is installed."""
        if not HAS_TF:
            self.skipTest("TensorFlow is not installed in current environment.")
        model = build_cnn_model(input_shape=(224, 224, 3), num_classes=38)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 38))
        self.assertGreater(len(model.layers), 10)

    def test_dummy_image_preprocessing(self):
        """Creates a temporary dummy leaf image and tests image reading and preprocessing."""
        dummy_img_path = config.OUTPUT_DIR / "test_dummy_leaf.jpg"
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        try:
            # Create a synthetic image array (300x300x3)
            synthetic_image = np.zeros((300, 300, 3), dtype=np.uint8)
            synthetic_image[:, :] = [34, 139, 34]  # Forest green color
            
            try:
                import cv2
                cv2.imwrite(str(dummy_img_path), synthetic_image)
            except ImportError:
                from PIL import Image
                img = Image.fromarray(synthetic_image)
                img.save(str(dummy_img_path))

            # Test load_image_cv2
            resized_img = load_image_cv2(str(dummy_img_path), target_size=(224, 224))
            self.assertEqual(resized_img.shape, (224, 224, 3))
            self.assertEqual(resized_img.dtype, np.uint8)

            # Test preprocess_single_image
            batch_tensor = preprocess_single_image(str(dummy_img_path), target_size=(224, 224))
            self.assertEqual(batch_tensor.shape, (1, 224, 224, 3))
            self.assertEqual(batch_tensor.dtype, np.float32)
            self.assertLessEqual(batch_tensor.max(), 1.0)
            self.assertGreaterEqual(batch_tensor.min(), 0.0)

        finally:
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)


if __name__ == "__main__":
    unittest.main()
