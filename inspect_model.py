import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import os

try:
    model_path = "models/best_gru.h5"
    if not os.path.exists(model_path):
        print("Model not found.")
    else:
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})
        print(f"Input Shape: {model.input_shape}")
        # Expected: (None, seq_len, n_features)
except Exception as e:
    print(f"Error: {e}")
