# rf_exploration.py
# Random Forest deep dive: tuning, evaluation, and benchmarking

import pandas as pd
import time
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Allow import from utils directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(os.path.join(project_root, "scripts", "utils"))
from benchmark_utils import create_benchmark_log, log_model_result, save_benchmark_log

# === Paths ===
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
models_dir = os.path.join(project_root, "models")
benchmark_file = os.path.join(project_root, "data", "rf_benchmark_log.csv")

# === Load data ===
df = pd.read_csv(data_path)
y = df['Price']
X = df.drop(columns=['Price'])

# === Scale numeric features ===
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save scaler
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, "rf_scaler.pkl"))

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize benchmark log ===
benchmark_log = create_benchmark_log()

# === Parameter grid to explore ===
param_grid = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 200, "max_depth": 10},
]

# === Run experiments ===
for params in param_grid:
    print(f"🔍 Training RandomForest with {params}")
    start = time.time()

    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()
    train_time = round(end - start, 2)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variant = f"n{params['n_estimators']}_d{params['max_depth']}"
    model_filename = os.path.join(models_dir, f"rf_{variant}.pkl")
    joblib.dump(model, model_filename)

    benchmark_log = log_model_result(
        benchmark_log,
        model_name="RandomForest",
        variant=variant,
        mae=mae,
        r2=r2,
        train_time=train_time,
        params=params,
        notes="RandomForest hyperparameter tuning"
    )

# === Save benchmark results ===
save_benchmark_log(benchmark_log, file_name="rf_benchmark_log.csv", log_dir=os.path.join(project_root, "data"))
print("✅ Random Forest exploration complete.")
