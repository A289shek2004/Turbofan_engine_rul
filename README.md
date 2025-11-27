# ✈️ Turbofan Engine RUL Prediction

## 📌 Project Overview

This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan jet engines using the NASA C-MAPSS dataset. It implements and evaluates multiple machine learning and deep learning models to estimate how many cycles an engine can operate before failure. The final solution is deployed as an interactive web application using **Streamlit**.

## 📊 Dataset

**Source:** NASA C-MAPSS (FD001–FD004 subsets)

Each dataset contains:
*   **Unit ID**: Engine identifier
*   **Cycle**: Operational cycle
*   **Operating conditions**: 3 settings
*   **Sensor readings**: 21 sensors (s1–s21)

**Labels:**
*   **RUL** = (Max cycle per engine – Current cycle)
*   For test sets, NASA provides ground-truth RUL values.

## ⚙️ Pipeline Workflow

### 1. Data Preprocessing
*   Loaded raw sensor data from `train_FDxxx.txt` and `test_FDxxx.txt`.
*   Normalized and smoothed sensor readings.
*   Created target labels for RUL prediction.

### 2. Feature Engineering
*   Time-series windowing for sequence learning (Sequence Length = 50).
*   Noise reduction techniques applied.
*   Selection of valid sensors based on variance (removing constant sensors).

### 3. Model Training & Evaluation
Implemented and compared:
*   **LSTM** (Long Short-Term Memory)
*   **GRU** (Gated Recurrent Unit)
*   **CNN-LSTM** (Hybrid model)

**Evaluation Metrics:**
*   RMSE (Root Mean Squared Error)
*   MAE (Mean Absolute Error)
*   NASA Scoring Function (penalizes late predictions more than early ones)

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8+
*   pip

### Installation
1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline
To train the models and generate results:
```bash
python run_pipeline.py
```
This will:
*   Process the data.
*   Train LSTM, GRU, and CNN-LSTM models.
*   Save the best models to the `models/` directory.
*   Save the scalers for data transformation.
*   Generate a results CSV in `outputs/`.

### Running the Application
To launch the interactive dashboard:
```bash
streamlit run app.py
```
The app allows you to:
*   Select a dataset (FD001-FD004) and model architecture.
*   Upload test data.
*   View RUL predictions and sensor trends.

## 📈 Results (FD001 Example)

| Model | Seq_len | Units | RMSE | NASA Score |
| :--- | :--- | :--- | :--- | :--- |
| **GRU** | 50 | 128 | ~11.47 | ~6234 |
| **LSTM** | 50 | 128 | ~12.47 | ~8375 |

**Key Learnings:**
*   Sequence models (LSTM, GRU) are highly effective for time-series degradation prediction.
*   **GRU** provided the most stable and accurate results for this dataset.

## 🛠️ Project Structure

*   `app.py`: Streamlit dashboard for interactive RUL prediction.
*   `run_pipeline.py`: Full pipeline (data preprocessing, training, evaluation).
*   `turbofan_rul_final.py`: Alternative pipeline script.
*   `models/`: Directory for saving trained models and scalers.
*   `outputs/`: Directory for saved results and plots.
*   `requirements.txt`: List of Python dependencies.
