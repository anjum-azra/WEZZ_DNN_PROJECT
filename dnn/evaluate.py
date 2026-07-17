"""
dnn/evaluate.py

Evaluate WEZ Deep Neural Network

Features
--------
✓ Load Trained Model
✓ Evaluate Test Dataset
✓ MAE
✓ RMSE
✓ R² Score
✓ Prediction CSV
✓ Prediction Plot
✓ Residual Plot
✓ Logging

Author: Anjum Azra
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras.models import load_model

from dnn.preprocess import DataPreprocessor
from utils.logger import logger


# ----------------------------------------------------------
# Prediction Plot
# ----------------------------------------------------------

def plot_predictions(actual, predicted):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.7,
    )

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        linewidth=2,
    )

    plt.xlabel("Actual Rmax")
    plt.ylabel("Predicted Rmax")
    plt.title("Predicted vs Actual")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/prediction_vs_actual.png",
        dpi=300,
    )

    plt.close()


# ----------------------------------------------------------
# Residual Plot
# ----------------------------------------------------------

def plot_residuals(actual, predicted):

    residuals = actual - predicted

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Predicted Rmax")
    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/residual_plot.png",
        dpi=300,
    )

    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    logger.info("Evaluation Started")

    os.makedirs("results", exist_ok=True)

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

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = load_model(
        "models/final_model.keras"
    )

    print()

    print("=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    predictions = model.predict(
        X_test,
        verbose=0,
    )

    predictions = predictions.flatten()

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    logger.info(
        "Evaluation | MAE=%.2f RMSE=%.2f R2=%.4f",
        mae,
        rmse,
        r2,
    )

    results = pd.DataFrame({

        "ActualRmax": y_test,

        "PredictedRmax": predictions,

        "AbsoluteError": np.abs(
            y_test - predictions
        ),

    })

    results.to_csv(

        "results/predictions.csv",

        index=False,

    )

    plot_predictions(
        y_test,
        predictions,
    )

    plot_residuals(
        y_test,
        predictions,
    )

    print()

    print("=" * 60)
    print("FILES GENERATED")
    print("=" * 60)

    print("✓ results/predictions.csv")
    print("✓ results/prediction_vs_actual.png")
    print("✓ results/residual_plot.png")

    print()

    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    logger.info("Evaluation Completed")


if __name__ == "__main__":

    main()