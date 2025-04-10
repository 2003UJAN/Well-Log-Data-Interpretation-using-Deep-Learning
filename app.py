# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and metadata
model = load_model("content/well_log_model.h5")
classes = np.load("content/facies_classes.npy", allow_pickle=True)

st.set_page_config(page_title="🛢️ Lithofacies Classifier", layout="centered")
st.title("🛢️ Well Log Lithofacies Classifier")
st.markdown("Enter the log values to predict the lithofacies class.")

# Input form
GR = st.slider("Gamma Ray (GR)", 40.0, 120.0, 80.0)
RHOB = st.slider("Bulk Density (RHOB)", 2.0, 3.0, 2.5)
NPHI = st.slider("Neutron Porosity (NPHI)", 0.2, 0.6, 0.4)
DT = st.slider("Sonic Travel Time (DT)", 90.0, 140.0, 115.0)
ILD = st.slider("Deep Resistivity (ILD)", 30.0, 70.0, 50.0)

if st.button("Predict Lithofacies"):
    input_data = np.array([[GR, RHOB, NPHI, DT, ILD]])
    scaled_data = (input_data - input_data.mean(axis=0)) / input_data.std(axis=0)
    scaled_data = scaled_data.reshape((scaled_data.shape[0], scaled_data.shape[1], 1))
    prediction = model.predict(scaled_data)
    predicted_class = facies_classes[np.argmax(prediction)]
    st.success(f"🏷️ Predicted Lithofacies: **{predicted_class}**")
