"""
Backend Inference script for predicting plant leaf disease from an input image.
Supports single image prediction via Command Line Interface (CLI).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.dataset import preprocess_single_image


def predict_leaf_disease(image_path: str, model_path: str = str(config.MODEL_PATH), top_k: int = 3):
    """
    Predicts the disease class of a given leaf image file.
    
    Args:
        image_path: Path to input leaf image.
        model_path: Path to trained Keras model file.
        top_k: Number of top probabilities to return.
        
    Returns:
        list of tuples: [(class_name, confidence_percentage), ...]
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python backend/train.py' first or place trained weights in models/ directory."
        )

    input_tensor = preprocess_single_image(image_path, target_size=config.IMG_SIZE)

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(input_tensor, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        class_name = config.CLASS_NAMES[idx] if idx < len(config.CLASS_NAMES) else f"Class_{idx}"
        confidence = float(predictions[idx]) * 100.0
        results.append((class_name, confidence))

    return results


def main():
    parser = argparse.ArgumentParser(description="Crop Disease Detection Inference CLI")
    parser.add_argument("--image", type=str, help="Path to input crop leaf image")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH), help="Path to trained model")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    
    args = parser.parse_args()

    if not args.image:
        print("Usage: python backend/predict.py --image <path_to_leaf_image.jpg>")
        print("\nExample:")
        print("  python backend/predict.py --image sample_leaf.jpg")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" CROP DISEASE DETECTION INFERENCE")
    print("=" * 60)
    print(f"Input Image : {args.image}")
    print(f"Model Path  : {args.model}")

    try:
        results = predict_leaf_disease(args.image, args.model, top_k=args.top_k)
        print("\nTop Predictions:")
        print("-" * 60)
        for rank, (cls_name, conf) in enumerate(results, 1):
            print(f"  {rank}. {cls_name:<45} [{conf:.2f}%]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[!] Prediction Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
