# turbofan_rul_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanSquaredError

# ✅ Set page config as the first Streamlit command
st.set_page_config(page_title="Turbofan Engine RUL Predictor", layout="wide")

# --------------------------
# Load Model + Scaler
# --------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "models/best_gru.h5",
            custom_objects={"mse": MeanSquaredError()}
        )
        scaler = joblib.load("models/scaler_gru.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model()

# --------------------------
# Utils
# --------------------------
def prepare_sequences(df, seq_len=50, sensor_cols=None):
    """Convert engine dataframe into model-ready sequence windows"""
    X, unit_ids = [], []
    for uid, u_df in df.groupby("unit"):
        data = u_df.sort_values("cycle")[sensor_cols].values
        if len(data) < seq_len:
            pad = np.repeat(data[0:1, :], seq_len - len(data), axis=0)
            data = np.vstack([pad, data])
        else:
            data = data[-seq_len:]
        X.append(data)
        unit_ids.append(uid)
    X = np.array(X)
    # scale
    n_samples, seq_len, n_feats = X.shape
    X_flat = X.reshape(-1, n_feats)
    X_scaled = scaler.transform(X_flat).reshape(n_samples, seq_len, n_feats)
    return X_scaled, unit_ids

def predict_rul(df, seq_len=50, sensor_cols=None):
    X, unit_ids = prepare_sequences(df, seq_len, sensor_cols)
    preds = model.predict(X).flatten()
    return pd.DataFrame({"unit": unit_ids, "Predicted_RUL": preds})

# --------------------------
# Streamlit UI
# --------------------------
st.title("🛠️ Turbofan Remaining Useful Life (RUL) Predictor")
st.markdown("""
Upload your CMAPSS engine sensor data file and get predictions for Remaining Useful Life (RUL).  
This app uses a **GRU/LSTM deep learning model** trained on NASA C-MAPSS datasets.  
""")
# --------------------------
# File Upload + Preprocessing + Prediction
# --------------------------
uploaded_file = st.file_uploader("📂 Upload engine sensor CSV/TXT", type=["csv", "txt"])

if uploaded_file:
    try:
        # --------------------------
        # Load dataset (try headers, else NASA default)
        # --------------------------
        try:
            df = pd.read_csv(uploaded_file, header=0, delim_whitespace=True)
        except Exception:
            df = pd.read_csv(uploaded_file)

        # If 'unit' missing → assign NASA default columns
        if "unit" not in df.columns:
            col_names = (
                ["unit", "cycle", "setting_1", "setting_2", "setting_3"] +
                [f"s{i}" for i in range(1, 22)]
            )
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=col_names)

        # Drop empty rows
        df = df.dropna().reset_index(drop=True)

        # Ensure correct types
        if "cycle" in df.columns:
            df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
            df = df.dropna(subset=["cycle"])
            df["cycle"] = df["cycle"].astype(int)

        st.subheader("📄 Preview of Uploaded Data")
        st.write(df.head())

        # --------------------------
        # Sidebar Filters
        # --------------------------
        st.sidebar.header("⚙️ Controls")

        # Sequence length
        seq_len = st.sidebar.slider("Sequence Length", 30, 100, 50)

        # Filter by unit
        if "unit" in df.columns:
            all_units = sorted(df["unit"].unique().tolist())
            selected_units = st.sidebar.multiselect("Select Engines (unit)", all_units, default=all_units)
            df = df[df["unit"].isin(selected_units)]

                # Filter by cycle range
        if not df.empty and "cycle" in df.columns:
            min_cycle, max_cycle = int(df["cycle"].min()), int(df["cycle"].max())
            if min_cycle < max_cycle:
                cycle_range = st.sidebar.slider("Cycle Range", min_cycle, max_cycle, (min_cycle, max_cycle))
                df = df[(df["cycle"] >= cycle_range[0]) & (df["cycle"] <= cycle_range[1])]
            else:
                st.sidebar.info(f"Cycle fixed at {min_cycle}")

        # Choose sensors
        all_sensors = [c for c in df.columns if c.startswith("s")]
        selected_sensors = st.sidebar.multiselect("Select Sensors", all_sensors, default=all_sensors)
        sensor_cols = selected_sensors if selected_sensors else all_sensors

        # Filter by operating conditions
        if {"setting_1", "setting_2", "setting_3"}.issubset(df.columns):
            st.sidebar.markdown("### Operating Condition Filters")
            for setting in ["setting_1", "setting_2", "setting_3"]:
                min_val, max_val = float(df[setting].min()), float(df[setting].max())
                if min_val < max_val:
                    chosen = st.sidebar.slider(f"{setting}", min_val, max_val, (min_val, max_val))
                    df = df[(df[setting] >= chosen[0]) & (df[setting] <= chosen[1])]
                else:
                    st.sidebar.info(f"{setting} fixed at {min_val}")

        # --------------------------
        # Prediction
        # --------------------------
        if st.button("🔮 Predict RUL"):
            if df.empty:
                st.warning("⚠️ No data available after applying filters. Please adjust filters.")
            else:
                results = predict_rul(df, seq_len=seq_len, sensor_cols=sensor_cols)
                st.success("✅ Predictions completed!")
                st.dataframe(results)

                # Download button
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name="rul_predictions.csv",
                    mime="text/csv"
                )

                # Visualization: bar chart
                st.subheader("📊 RUL Predictions per Engine")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(results["unit"], results["Predicted_RUL"], color="skyblue")
                ax.set_xlabel("Engine Unit")
                ax.set_ylabel("Predicted RUL (cycles)")
                ax.set_title("Predicted Remaining Useful Life")
                st.pyplot(fig)

                # Line chart (RUL curve for one sample engine)
                if "cycle" in df.columns and not results.empty:
                    st.subheader("📉 Example RUL Curve (sample engine)")
                    sample_id = results["unit"].iloc[0]
                    sample_df = df[df["unit"] == sample_id].sort_values("cycle")

                    if not sample_df.empty:
                        true_cycles = np.arange(len(sample_df))
                        predicted_rul_start = results.loc[results["unit"] == sample_id, "Predicted_RUL"].values[0]
                        rul_curve = np.linspace(predicted_rul_start, 0, len(sample_df))

                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        ax2.plot(true_cycles, rul_curve, label="Predicted RUL", color="red")
                        ax2.set_xlabel("Cycle")
                        ax2.set_ylabel("RUL")
                        ax2.legend()
                        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("👆 Upload a CMAPSS-like dataset to start predictions.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Internship Project Deployment")
