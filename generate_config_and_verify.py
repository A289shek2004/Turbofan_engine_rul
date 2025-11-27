import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import json
import os

def load_cmapss(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1,4)]
    sensor_cols = [f"s{i}" for i in range(1, 22)] # 21 sensors
    df.columns = cols + sensor_cols
    return df

def drop_constant_sensors(train_df):
    sensors = [c for c in train_df.columns if c.startswith("s")]
    keep = [c for c in sensors if train_df[c].std() > 0]
    return keep

def main():
    print("Loading model...")
    try:
        model = tf.keras.models.load_model("models/best_gru.h5", custom_objects={"mse": MeanSquaredError()})
        input_shape = model.input_shape
        print(f"Model Input Shape: {input_shape}")
        
        seq_len = input_shape[1]
        n_features = input_shape[2]
        
        print(f"Detected: seq_len={seq_len}, n_features={n_features}")
        
        print("Loading training data to verify features...")
        train_df = load_cmapss(r"E:\Turbofan engine project\train_FD001.txt")
        sensor_cols = drop_constant_sensors(train_df)
        
        print(f"Found {len(sensor_cols)} non-constant sensors in data.")
        
        if len(sensor_cols) == n_features:
            print("✅ Feature count matches!")
            config = {
                "sensor_cols": sensor_cols,
                "seq_len": int(seq_len),
                "model_type": "GRU"
            }
            with open("models/model_config.json", "w") as f:
                json.dump(config, f, indent=2)
            print("Saved models/model_config.json")
        else:
            print(f"❌ Mismatch! Model expects {n_features} features, but data has {len(sensor_cols)}.")
            print("You may need to retrain the model.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
