import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config
DATA_DIR = "."
OUT_DIR = "outputs"
MODELS_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SEQ_LEN = 50
RUL_CAP = 125
BATCH_SIZE = 128
EPOCHS = 20  # Increased for better convergence
PATIENCE = 5

def load_data(dataset):
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    
    def _load_file(path):
        try:
            # Check first line for header
            with open(path, 'r') as f:
                head = f.readline()
            if "unit" in head.lower() or "s1" in head.lower():
                return pd.read_csv(path, sep=r"\s+", header=0)
            return pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return pd.DataFrame()

    train = _load_file(os.path.join(DATA_DIR, f"train_{dataset}.txt"))
    test = _load_file(os.path.join(DATA_DIR, f"test_{dataset}.txt"))
    rul = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{dataset}.txt"), sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul["RUL"].values

def process_data(train_df, test_df):
    # Calculate RUL for train
    max_cycle = train_df.groupby("unit")["cycle"].transform("max")
    train_df["RUL"] = max_cycle - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)
    
    # Identify valid sensors (exclude constants)
    sensor_cols = [c for c in train_df.columns if c.startswith("s")]
    # Filter sensors with std > 0.01 (handles FD001/FD003 constants)
    valid_sensors = [c for c in sensor_cols if train_df[c].std() > 0.01]
    
    # Scale
    scaler = MinMaxScaler()
    train_df[valid_sensors] = scaler.fit_transform(train_df[valid_sensors])
    test_df[valid_sensors] = scaler.transform(test_df[valid_sensors])
    
    return train_df, test_df, valid_sensors, scaler

def gen_sequence(df, seq_len, cols):
    X, y = [], []
    for _, group in df.groupby("unit"):
        data = group[cols].values
        rul = group["RUL"].values
        # Create sequences
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(rul[i+seq_len])
    return np.array(X), np.array(y)

def gen_test_sequence(df, seq_len, cols):
    X, unit_ids = [], []
    for uid, group in df.groupby("unit"):
        data = group[cols].values
        if len(data) >= seq_len:
            X.append(data[-seq_len:])
            unit_ids.append(uid)
        else:
            pad = np.repeat(data[0:1, :], seq_len - len(data), axis=0)
            X.append(np.vstack([pad, data]))
            unit_ids.append(uid)
    return np.array(X), unit_ids

# Model Builders
def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        GRU(64),
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

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1))

if __name__ == "__main__":
    results = []
    
    for fd in DATASETS:
        print(f"\n{'='*30}\nProcessing {fd}\n{'='*30}")
        try:
            # Load & Process
            train_df, test_df, rul_true = load_data(fd)
            train_df, test_df, sensors, scaler = process_data(train_df, test_df)
            
            print(f"Sensors ({len(sensors)}): {sensors}")
            
            # Save Scaler
            joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{fd}.pkl"))
            
            # Sequences
            X_train, y_train = gen_sequence(train_df, SEQ_LEN, sensors)
            X_test, _ = gen_test_sequence(test_df, SEQ_LEN, sensors)
            print(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Train Models
            models = {
                "LSTM": build_lstm,
                "GRU": build_gru,
                "CNN-LSTM": build_cnn_lstm
            }
            
            for name, builder in models.items():
                print(f"Training {name}...")
                model = builder((SEQ_LEN, len(sensors)))
                es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
                
                model.fit(
                    X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[es],
                    verbose=1
                )
                
                # Eval
                preds = model.predict(X_test, verbose=0).flatten()
                rmse = np.sqrt(mean_squared_error(rul_true, preds))
                score = nasa_score(rul_true, preds)
                print(f"{name} -> RMSE: {rmse:.2f}, NASA: {score:.2f}")
                
                results.append({"Dataset": fd, "Model": name, "RMSE": rmse, "NASA": score})
                
                # Save Model
                model.save(os.path.join(MODELS_DIR, f"model_{fd}_{name}.h5"))
                
        except Exception as e:
            print(f"Error on {fd}: {e}")
            
    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "final_results.csv"), index=False)
    print("\nPipeline Complete.")
    print(res_df)
