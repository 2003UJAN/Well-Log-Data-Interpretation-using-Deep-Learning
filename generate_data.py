# generate_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    features = ['GR', 'RHOB', 'NPHI', 'DT', 'ILD']
    target = 'Facies'

    df = df.dropna(subset=features + [target])
    
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    return X_scaled, y_encoded, scaler, encoder
