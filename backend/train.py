"""
Backend Model training pipeline script.
Loads dataset generators, builds CNN model, manages callbacks, trains the network,
saves history plots, and serializes trained model weights.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.dataset import get_data_generators
from backend.model import build_cnn_model


def plot_training_history(history, save_path: Path):
    """Generates and saves training loss and accuracy plots."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#2b5c8f", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#d95f02", linewidth=2)
    axes[0].set_title("Model Accuracy Over Epochs", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    axes[1].plot(epochs_range, loss, label="Training Loss", color="#2b5c8f", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#d95f02", linewidth=2)
    axes[1].set_title("Model Loss Over Epochs", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to: {save_path}")


def train():
    """Executes the complete training workflow."""
    print("======================================================================")
    print(" CROP DISEASE DETECTION - TRAINING PIPELINE")
    print("======================================================================")

    # Ensure output & model directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/4] Loading dataset generators...")
    try:
        train_gen, val_gen = get_data_generators(
            data_dir=config.DATASET_DIR,
            batch_size=config.BATCH_SIZE,
            img_size=config.IMG_SIZE
        )
    except Exception as e:
        print(f"\nDataset error: {e}")
        print("Please check data/README.md for dataset placement instructions.")
        sys.exit(1)

    print(f"      Train Samples: {train_gen.samples} | Validation Samples: {val_gen.samples}")
    print(f"      Total Target Classes: {train_gen.num_classes}")

    print("\n[2/4] Building CNN model architecture...")
    model = build_cnn_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=train_gen.num_classes
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n[3/4] Starting CNN model training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    print("\n[4/4] Saving model history and final evaluation...")
    plot_training_history(history, config.HISTORY_PLOT_PATH)

    val_loss, val_acc, val_top3 = model.evaluate(val_gen)
    print("\n======================================================================")
    print(" FINAL VALIDATION METRICS")
    print("======================================================================")
    print(f" Validation Loss      : {val_loss:.4f}")
    print(f" Validation Accuracy  : {val_acc * 100:.2f}%")
    print(f" Top-3 Categorical Acc: {val_top3 * 100:.2f}%")
    print(f" Saved Model Path     : {config.MODEL_PATH}")
    print("======================================================================")


if __name__ == "__main__":
    train()
