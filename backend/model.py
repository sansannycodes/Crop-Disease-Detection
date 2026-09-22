"""
Backend Convolutional Neural Network (CNN) architecture definition module.
Builds a multi-layer CNN for 38-class crop disease image classification using Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from backend import config


def build_cnn_model(input_shape: tuple = config.INPUT_SHAPE,
                    num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """
    Constructs and compiles a Convolutional Neural Network (CNN) for leaf disease classification.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier Block
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")]
    )

    return model


if __name__ == "__main__":
    cnn_model = build_cnn_model()
    cnn_model.summary()
