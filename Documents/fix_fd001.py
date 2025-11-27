import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Config
DATA_DIR = "."
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
SEQ_LEN = 50
RUL_CAP = 125
BATCH_SIZE = 128
EPOCHS = 10 

def load_data():
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    train = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"), sep=r"\s+", header=None, names=cols)
    return train

def process_data(df):
    # RUL
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    
    # Sensors (FD001 has constants, we must use same logic as app or save scaler correctly)
    # The scaler_FD001.pkl already exists, let's load it to see what features it expects
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_FD001.pkl"))
    
    # We need to identify which columns were used. 
    # Since we can't easily inspect the scaler's feature names (it's a numpy array usually),
    # we will re-derive the non-constant sensors which is standard for FD001.
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    valid_sensors = [c for c in sensor_cols if df[c].std() > 0.01]
    
    # Check if count matches scaler
    if len(valid_sensors) != scaler.n_features_in_:
        print(f"Warning: Feature count mismatch. Scaler expects {scaler.n_features_in_}, found {len(valid_sensors)}")
        # If mismatch, we might need to retrain scaler too to be safe.
        print("Retraining scaler...")
        scaler = MinMaxScaler()
        df[valid_sensors] = scaler.fit_transform(df[valid_sensors])
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_FD001.pkl"))
    else:
        df[valid_sensors] = scaler.transform(df[valid_sensors])
        
    return df, valid_sensors

def gen_sequence(df, seq_len, cols):
    X, y = [], []
    for _, group in df.groupby("unit"):
        data = group[cols].values
        rul = group["RUL"].values
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(rul[i+seq_len])
    return np.array(X), np.array(y)

def build_gru(input_shape):
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("Fixing FD001 Models...")
    df = load_data()
    df, sensors = process_data(df)
    X, y = gen_sequence(df, SEQ_LEN, sensors)
    print(f"Data shape: {X.shape}")
    
    # Train GRU
    print("Training GRU...")
    gru = build_gru((SEQ_LEN, len(sensors)))
    gru.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    gru.save(os.path.join(MODELS_DIR, "model_FD001_GRU.h5"))
    
    # Train CNN-LSTM
    print("Training CNN-LSTM...")
    cnn = build_cnn_lstm((SEQ_LEN, len(sensors)))
    cnn.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    cnn.save(os.path.join(MODELS_DIR, "model_FD001_CNN-LSTM.h5"))
    
    print("Done.")
