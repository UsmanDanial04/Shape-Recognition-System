# train_model.py
"""
CNN definition and training for the shape classifier.

Prompt 7 implementation.

Usage:
    python train_model.py

Requires:
    tensorflow  (Python ≤ 3.12)
    data/X_train.npy  … (from prepare_dataset.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Guard against running on Python 3.14 where TF is unavailable
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
except ImportError:
    raise SystemExit(
        "TensorFlow is required for training.  "
        "Install it in a Python ≤ 3.12 environment:\n"
        "    pip install tensorflow"
    )

from sklearn.metrics import confusion_matrix
from config import SHAPE_CLASSES

DATA_DIR   = "data"
MODELS_DIR = "models"
NUM_CLASSES = len(SHAPE_CLASSES)

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> tf.keras.Model:
    """Return a compiled Sequential CNN.

    Architecture:
        Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
        Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
        Flatten
        Dense(128, ReLU) → Dropout(0.5)
        Dense(num_classes, softmax)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(28, 28, 1), padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="shape_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_history(history):
    """Plot and save training / validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14)

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Loss
    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "training_history.png")
    plt.savefig(out, dpi=100)
    plt.show()
    print(f"Training curves saved to {out}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save a confusion-matrix heatmap."""
    cm  = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=100)
    plt.show()
    print(f"Confusion matrix saved to {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    print("Loading data …")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"  Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    # ---- Build & train ----------------------------------------------------
    model = build_model(NUM_CLASSES)

    checkpoint_path = os.path.join(MODELS_DIR, "shape_classifier.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
    )

    # ---- Evaluate ---------------------------------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy : {test_acc * 100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")

    # ---- Plots ------------------------------------------------------------
    plot_history(history)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    plot_confusion_matrix(y_test, y_pred, SHAPE_CLASSES)

    print(f"\nBest model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
