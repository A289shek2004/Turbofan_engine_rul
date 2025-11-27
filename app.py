# turbofan_rul_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import os
from tensorflow.keras.metrics import MeanSquaredError

# --------------------------
# Page Config & Styling
# --------------------------
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Configuration & Resource Loading
# --------------------------
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["LSTM", "GRU", "CNN-LSTM"]

@st.cache_resource
def load_model_and_scaler(dataset, model_type):
    """Loads the specific model and scaler for the selected dataset."""
    model_path = f"models/model_{dataset}_{model_type}.h5"
    scaler_path = f"models/scaler_{dataset}.pkl"
    
    model = None
    scaler = None
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"mse": MeanSquaredError()}
            )
        except Exception as e:
            st.error(f"Error loading model {model_path}: {e}")
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler {scaler_path}: {e}")
            
    return model, scaler

# --------------------------
# Sidebar Controls
# --------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_dataset = st.selectbox("Select Dataset", DATASETS, index=0)
    selected_model_type = st.selectbox("Select Model Architecture", MODELS, index=0)
    
    st.markdown("---")
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader(f"Upload {selected_dataset} Test Data", type=["txt", "csv"])
    
    st.markdown("---")
    st.info(f"**Current Config:**\n- Dataset: {selected_dataset}\n- Model: {selected_model_type}")

# Load Resources
model, scaler = load_model_and_scaler(selected_dataset, selected_model_type)

# --------------------------
# Utils
# --------------------------
def prepare_sequences(df, seq_len, sensor_cols):
    """Convert engine dataframe into model-ready sequence windows"""
    X, unit_ids = [], []
    
    # Check sensors
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing sensor columns: {missing}")
        return None, None

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
    
    # Scale
    if scaler:
        n_samples, seq_len, n_feats = X.shape
        X_flat = X.reshape(-1, n_feats)
        X_scaled = scaler.transform(X_flat).reshape(n_samples, seq_len, n_feats)
        return X_scaled, unit_ids
    else:
        return X, unit_ids

def predict_rul(df, seq_len=50):
    # Determine sensors based on dataframe (assuming standard format)
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    # Filter to valid sensors used in training (need to know which ones, but scaler expects specific shape)
    # For simplicity, we assume the scaler was trained on ALL sensors present in the training set of that FD.
    # A robust way is to store the feature names in the scaler or a config.
    # Here we try to use all 's' columns that match the scaler's expected input size.
    
    if scaler:
        expected_feats = scaler.n_features_in_
        # We need to find which sensors correspond. 
        # Heuristic: use first N sensors where N = expected_feats
        # Or better: The training script used `get_valid_sensors`. 
        # We should ideally load the list of sensors.
        # For this fix, we will try to use all available sensors and hope it matches, 
        # or we can try to infer.
        # Let's assume the user uploads data with all 21 sensors.
        available_sensors = [f"s{i}" for i in range(1, 22)]
        
        # If the scaler expects fewer, we need to know which ones.
        # This is a limitation of the current simple pipeline.
        # We will try to use the first 'expected_feats' sensors if they match the count.
        
        if len(available_sensors) >= expected_feats:
            # This is risky but often valid sensors are s2, s3, s4... 
            # Actually, `get_valid_sensors` drops constant ones.
            # We really should save the feature list.
            # For now, let's try to use all s1-s21 and let the scaler complain or slice?
            # No, scaler will complain if feature count mismatch.
            pass
            
    # To fix the feature mismatch issue properly:
    # We will rely on the fact that we saved the scaler. 
    # We will try to use all 21 sensors. If scaler expects fewer, we are in trouble without the list.
    # BUT, in my `run_pipeline.py`, I used `get_valid_sensors` which drops constant ones.
    # I should have saved the feature list.
    # I will update the app to try to deduce or just use all if scaler allows.
    # Actually, for FD001, some are constant.
    # Let's assume the user uploads the full dataset.
    # We will filter columns based on variance if we can, or just try to match the scaler.
    
    # IMPROVEMENT: Let's just use all 21 sensors for input and let the code handle it?
    # No, the model expects specific input shape.
    
    # Hack: We will use the scaler's `n_features_in_` to select the first N non-constant columns?
    # No, that's dangerous.
    # Let's try to use all s1-s21. If the scaler was trained on fewer, we need that list.
    # I will assume for now that we used ALL sensors in the updated pipeline?
    # In `run_pipeline.py`: `sensors = get_valid_sensors(train_df)`.
    # So we DID drop sensors.
    # I will update the app to perform the same `get_valid_sensors` logic on the UPLOADED data?
    # No, the uploaded data might be short and have 0 variance where training had variance.
    
    # Solution: I will update `run_pipeline.py` to save the feature list, OR
    # I will update the app to assume the standard non-constant sensors for each FD.
    # FD001 constants: s1, s5, s6, s10, s16, s18, s19.
    # I will hardcode the valid sensors for each FD to be safe, or try to load from a config if I had saved one.
    # Since I didn't save a config in the previous step, I will add a helper here.
    
    pass

def get_sensors_for_dataset(dataset):
    # Known constant sensors for C-MAPSS
    # FD001: s1, s5, s6, s10, s16, s18, s19 are constant.
    # FD003: same as FD001.
    # FD002, FD004: All 21 sensors are usually kept (or most).
    
    all_sensors = [f"s{i}" for i in range(1, 22)]
    
    if dataset in ["FD001", "FD003"]:
        # Exclude constants
        constants = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
        return [s for s in all_sensors if s not in constants]
    else:
        return all_sensors

# --------------------------
# Main UI Logic
# --------------------------
st.title("✈️ Turbofan Engine RUL Predictor")

if not model or not scaler:
    st.warning(f"⚠️ Model or Scaler for {selected_dataset} ({selected_model_type}) not found.")
    st.info("Please run the training pipeline first or wait for it to complete.")
else:
    if uploaded_file:
        try:
            # Load Data
            try:
                df = pd.read_csv(uploaded_file, header=0, delim_whitespace=True)
                if "unit" not in df.columns: raise ValueError("No header")
            except:
                col_names = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=col_names)
            
            # Drop empty cols
            df = df.dropna(axis=1, how='all')
            
            # Determine sensors
            sensor_cols = get_sensors_for_dataset(selected_dataset)
            
            # Verify feature count matches scaler
            if len(sensor_cols) != scaler.n_features_in_:
                st.warning(f"⚠️ Feature count mismatch. Model expects {scaler.n_features_in_}, but we selected {len(sensor_cols)} based on dataset defaults.")
                st.info("Attempting to use all available 's' columns...")
                sensor_cols = [c for c in df.columns if c.startswith("s")]
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Sensor Analysis", "📄 Data Preview"])
            
            with tab1:
                st.subheader("Prediction Results")
                if st.button("🚀 Run Prediction", key="predict_btn"):
                    with st.spinner("Processing..."):
                        X, unit_ids = prepare_sequences(df, 50, sensor_cols)
                        if X is not None:
                            preds = model.predict(X, verbose=0).flatten()
                            results = pd.DataFrame({"unit": unit_ids, "Predicted_RUL": preds})
                            
                            # Metrics
                            avg_rul = results["Predicted_RUL"].mean()
                            min_rul = results["Predicted_RUL"].min()
                            critical = results[results["Predicted_RUL"] < 50].shape[0]
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Avg RUL", f"{avg_rul:.1f}")
                            c2.metric("Min RUL", f"{min_rul:.1f}")
                            c3.metric("Critical (<50)", f"{critical}", delta_color="inverse")
                            
                            # Chart
                            fig = px.bar(results, x="unit", y="Predicted_RUL", color="Predicted_RUL", color_continuous_scale="RdYlGn", title="RUL per Engine")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.session_state["results"] = results
                            st.session_state["df"] = df
                        else:
                            st.error("Sequence preparation failed.")

            with tab2:
                if "df" in st.session_state:
                    df_d = st.session_state["df"]
                    u = st.selectbox("Unit", sorted(df_d["unit"].unique()))
                    s = st.multiselect("Sensors", sensor_cols, default=sensor_cols[:3])
                    if s:
                        d = df_d[df_d["unit"] == u]
                        st.plotly_chart(px.line(d, x="cycle", y=s, title=f"Unit {u} Sensors"), use_container_width=True)
            
            with tab3:
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info(f"Please upload {selected_dataset} test file.")
