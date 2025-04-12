import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model and class labels
model = load_model("content/well_log_model.h5")
facies_classes = np.load("content/facies_classes.npy", allow_pickle=True).tolist()

st.set_page_config(page_title="Well Log Facies Predictor", layout="centered")

# UI Design
st.title("🛢️ Well Log Facies Classification using Deep Learning")
st.markdown("""
Welcome to the **Automated Lithofacies Classifier** powered by **1D CNN + GRU**.
Upload a well log sample CSV file with these logs: `GR`, `RHOB`, `NPHI`, `PE`, `DT` and get the predicted **facies**.
""")

uploaded_file = st.file_uploader("📤 Upload Well Log Sample CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['GR', 'RHOB', 'NPHI', 'PE', 'DT']

        if all(col in df.columns for col in required_columns):
            st.success("✅ Valid File Uploaded!")

            # Normalize input
            X = df[required_columns].values
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
            X = np.expand_dims(X, axis=0)

            prediction = model.predict(X)

            if prediction.shape[1] == len(facies_classes):
                predicted_class = facies_classes[np.argmax(prediction)]
                st.markdown(f"### 🧠 Predicted Lithofacies: `{predicted_class}`")
            else:
                st.error("❌ Model output shape does not match number of facies classes.")

        else:
            st.error(f"❌ Missing required columns! Please include: {', '.join(required_columns)}")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | © 2025")
