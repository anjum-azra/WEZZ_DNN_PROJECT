"""
app.py

WEZ Deep Neural Network
Streamlit Application

Features
--------
✓ Modern Web Interface
✓ Predict Maximum Launch Range
✓ Model Loading
✓ Scaler Loading
✓ Error Handling

Author: Anjum Azra
"""

import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(

    page_title="WEZ DNN Predictor",

    page_icon="🚀",

    layout="wide",

)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_resources():

    model = load_model("models/final_model.keras")

    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


model, scaler = load_resources()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚀 Weapon Engagement Zone (WEZ) Predictor")

st.write(
    """
Predict the **Maximum Launch Range (Rmax)** using a
Deep Neural Network trained on simulated missile
engagement data.
"""
)

st.divider()

# --------------------------------------------------
# Input Columns
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    shooter_altitude = st.number_input(

        "Shooter Altitude (m)",

        min_value=0.0,

        value=5000.0,

    )

    shooter_velocity = st.number_input(

        "Shooter Velocity (m/s)",

        min_value=0.0,

        value=300.0,

    )

    shooter_pitch = st.number_input(

        "Shooter Pitch (deg)",

        value=0.0,

    )

with col2:

    target_altitude = st.number_input(

        "Target Altitude (m)",

        min_value=0.0,

        value=6000.0,

    )

    target_velocity = st.number_input(

        "Target Velocity (m/s)",

        min_value=0.0,

        value=250.0,

    )

    target_heading = st.number_input(

        "Target Heading (deg)",

        value=180.0,

    )

    target_offboresight = st.number_input(

        "Target Off-Boresight (deg)",

        value=30.0,

    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Maximum Launch Range"):

    sample = np.array([[

        shooter_altitude,

        shooter_velocity,

        shooter_pitch,

        target_altitude,

        target_velocity,

        target_heading,

        target_offboresight,

    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(

        sample,

        verbose=0,

    )

    rmax = prediction[0][0]

    st.success("Prediction Completed")

    st.metric(

        "Maximum Launch Range",

        f"{rmax:,.2f} m",

    )

    st.metric(

        "Maximum Launch Range",

        f"{rmax/1000:.2f} km",

    )

st.divider()

st.caption(
    "WEZ Deep Neural Network | Developed by Anjum Azra"
)