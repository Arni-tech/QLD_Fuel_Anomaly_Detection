# mlp_exploration.py
# Stable MLP regressor with target scaling, safe inverse transform, and regularization

import pandas as pd
import time
import os
import sys
import joblib
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# === Path setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(os.path.join(project_root, "scripts", "utils"))
from benchmark_utils import create_benchmark_log, log_model_result, save_benchmark_log

# === Paths ===
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(data_path)
y = df['Price']
X = df.drop(columns=['Price'])

# === Scale features ===
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
x_scaler = StandardScaler()
X[numeric_cols] = x_scaler.fit_transform(X[numeric_cols])
X = X.dropna()
y = y.loc[X.index]

# === Scale target ===
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
joblib.dump(x_scaler, os.path.join(models_dir, "mlp_x_scaler.pkl"))
joblib.dump(y_scaler, os.path.join(models_dir, "mlp_y_scaler.pkl"))

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# === Benchmark log ===
benchmark_log = create_benchmark_log()

# === Configs to test ===
configs = [
    {"hidden_layer_sizes": (32,), "early_stopping": True, "alpha": 0.001},
    {"hidden_layer_sizes": (32, 16), "early_stopping": True, "alpha": 0.001},
    {"hidden_layer_sizes": (64, 32), "early_stopping": True, "alpha": 0.0005},
]

# === Train/evaluate loop ===
for cfg in configs:
    print(f"🔍 Training MLP with config: {cfg}")
    start = time.time()

    model = MLPRegressor(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        early_stopping=cfg["early_stopping"],
        alpha=cfg["alpha"],
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    # Sanity check before inverse_transform
    if np.any(np.isnan(y_pred_scaled)) or np.any(np.abs(y_pred_scaled) > 100):
        print("⚠️ Skipping inverse transform — model diverged.")
        mae = float('inf')
        r2 = float('-inf')
    else:
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

    end = time.time()
    train_time = round(end - start, 2)

    variant = f"{cfg['hidden_layer_sizes']}_es{cfg['early_stopping']}_a{cfg['alpha']}"
    model_filename = os.path.join(models_dir, f"mlp_{variant}.pkl")
    joblib.dump(model, model_filename)

    benchmark_log = log_model_result(
        benchmark_log,
        model_name="MLP",
        variant=variant,
        mae=mae,
        r2=r2,
        train_time=train_time,
        params=cfg,
        notes="Stable MLP with target scaling and divergence guard"
    )

# === Save results ===
save_benchmark_log(benchmark_log, file_name="mlp_benchmark_log.csv", log_dir=os.path.join(project_root, "data"))
print("✅ MLP exploration complete.")
