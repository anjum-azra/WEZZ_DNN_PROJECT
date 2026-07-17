"""
dnn/train.py

Train the WEZ Deep Neural Network

Features
--------
✓ Early Stopping
✓ Model Checkpoint
✓ Reduce Learning Rate
✓ TensorBoard
✓ Model Versioning
✓ Training History Plots
✓ Logging

Author: Anjum Azra
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from dnn.preprocess import DataPreprocessor
from dnn.model import build_model

from utils.logger import logger


# ----------------------------------------------------------
# Plot Training History
# ----------------------------------------------------------

def plot_history(history):

    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # Loss
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2,
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/training_loss.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------
    # MAE
    # -----------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["mae"],
        label="Training MAE",
        linewidth=2,
    )

    plt.plot(
        history.history["val_mae"],
        label="Validation MAE",
        linewidth=2,
    )

    plt.title("Training MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/training_mae.png",
        dpi=300,
    )

    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    logger.info("Training Started")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    processor = DataPreprocessor(
        "data/wez_dataset.csv"
    )

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = processor.process()

    print(f"Training Samples   : {len(X_train)}")
    print(f"Validation Samples : {len(X_valid)}")
    print(f"Testing Samples    : {len(X_test)}")

    logger.info(
        "Dataset Loaded | Train=%d Validation=%d Test=%d",
        len(X_train),
        len(X_valid),
        len(X_test),
    )

    print()

    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = build_model()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    versioned_model = f"models/wez_{timestamp}.keras"

    callbacks = [

        EarlyStopping(

            monitor="val_loss",

            patience=20,

            restore_best_weights=True,

            verbose=1,

        ),

        ModelCheckpoint(

            filepath="models/wez_dnn.keras",

            monitor="val_loss",

            save_best_only=True,

            verbose=1,

        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=10,

            min_lr=1e-6,

            verbose=1,

        ),

        TensorBoard(

            log_dir="logs/tensorboard",

            histogram_freq=1,

            write_graph=True,

        ),

    ]

    print()

    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_valid, y_valid),

        epochs=200,

        batch_size=32,

        callbacks=callbacks,

        verbose=1,

    )

    print()

    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    loss, mae = model.evaluate(

        X_test,

        y_test,

        verbose=0,

    )

    print(f"Test Loss : {loss:.4f}")
    print(f"Test MAE  : {mae:.4f}")

    logger.info(
        "Training Finished | Loss=%.4f MAE=%.4f",
        loss,
        mae,
    )

    print()

    print("=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    # Final model
    model.save(
        "models/final_model.keras"
    )

    # Timestamped model
    model.save(
        versioned_model
    )

    plot_history(history)

    print()

    print("Generated Files")

    print("---------------------------")

    print("✓ models/wez_dnn.keras")
    print("✓ models/final_model.keras")
    print(f"✓ {versioned_model}")

    print("✓ results/training_loss.png")
    print("✓ results/training_mae.png")

    print("✓ logs/tensorboard/")

    print()

    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    logger.info("Training Completed Successfully")


if __name__ == "__main__":

    main()