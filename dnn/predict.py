"""
dnn/predict.py

Predict WEZ Maximum Launch Range

Features
--------
✓ Load Trained Model
✓ Load Scaler
✓ User Input
✓ Predict Rmax

Author: Anjum Azra
"""

import joblib
import numpy as np

from tensorflow.keras.models import load_model

from utils.logger import logger


MODEL_PATH = "models/final_model.keras"
SCALER_PATH = "models/scaler.pkl"


def get_float(prompt):

    while True:

        try:

            return float(input(prompt))

        except ValueError:

            print("Invalid input. Please enter a numeric value.")


def get_inputs():

    print("\nEnter Engagement Parameters\n")

    shooter_altitude = get_float("Shooter Altitude (m): ")
    shooter_velocity = get_float("Shooter Velocity (m/s): ")
    shooter_pitch = get_float("Shooter Pitch (deg): ")

    target_altitude = get_float("Target Altitude (m): ")
    target_velocity = get_float("Target Velocity (m/s): ")
    target_heading = get_float("Target Heading (deg): ")
    target_offboresight = get_float("Target Off-Boresight (deg): ")

    return np.array([[
        shooter_altitude,
        shooter_velocity,
        shooter_pitch,
        target_altitude,
        target_velocity,
        target_heading,
        target_offboresight,
    ]])


def main():

    logger.info("Prediction Started")

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    sample = get_inputs()

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(
        sample_scaled,
        verbose=0,
    )

    rmax = prediction[0][0]

    print()

    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Predicted Maximum Launch Range : {rmax:.2f} meters")
    print(f"                              : {rmax/1000:.2f} km")

    logger.info(
        "Prediction Completed | Rmax = %.2f",
        rmax,
    )


if __name__ == "__main__":

    main()