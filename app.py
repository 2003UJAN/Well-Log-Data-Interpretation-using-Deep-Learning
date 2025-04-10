# app.py

import streamlit as st
import pandas as pd
import numpy as np
from generate_data import load_and_preprocess
from model import build_model
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Well Log Lithofacies Predictor", layout="wide")

st.title("🛢️ Well Log Lithofacies Predictor")
st.markdown("Upload your well log data CSV to classify lithofacies using our deep learning model (CNN + GRU).")

uploaded_file = st.file_uploader("Upload Well Log CSV", type=["csv"])

if uploaded_file:
    X, y, scaler, encoder = load_and_preprocess(uploaded_file)
    st.success("✅ Data loaded and preprocessed.")

    with st.spinner("Training model..."):
        model = build_model(input_shape=X.shape[1:], num_classes=len(np.unique(y)))
        history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    st.success("✅ Model trained.")

    # Predictions
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=1)
    decoded_preds = encoder.inverse_transform(pred_labels)

    df_results = pd.read_csv(uploaded_file)
    df_results['Predicted_Facies'] = decoded_preds

    st.subheader("📊 Results Sample")
    st.dataframe(df_results[['Depth', 'GR', 'RHOB', 'NPHI', 'DT', 'ILD', 'Predicted_Facies']].head())

    # Visualization
    st.subheader("📈 Prediction Distribution")
    fig, ax = plt.subplots()
    pd.Series(decoded_preds).value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Predicted Facies Distribution")
    st.pyplot(fig)

    st.download_button("📥 Download Predictions", df_results.to_csv(index=False), "predicted_facies.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to begin.")
