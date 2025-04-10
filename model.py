# model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(file_path="/content/well_log_sample.csv"):
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
    return X_scaled, y_encoded, encoder, scaler, encoder.classes_

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, encoder, scaler, classes = load_data()
    model = build_model(X.shape[1:], num_classes=len(np.unique(y)))
    checkpoint = ModelCheckpoint("/content/well_log_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint], verbose=1)

    print("✅ Model saved at /content/well_log_model.h5")
    np.save("/content/facies_classes.npy", classes)
    return scaler

if __name__ == "__main__":
    train_model()
