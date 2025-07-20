# knn_exploration.py
# K-Nearest Neighbors regression: tuning neighbors, evaluation, and benchmarking

import pandas as pd
import time
import os
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Allow import from utils directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(os.path.join(project_root, "scripts", "utils"))
from benchmark_utils import create_benchmark_log, log_model_result, save_benchmark_log

# === Paths ===
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
models_dir = os.path.join(project_root, "models")

# === Load and prepare data ===
df = pd.read_csv(data_path)
y = df['Price']
X = df.drop(columns=['Price'])  # Assumes TransactionDateutc already removed

# === Scale features ===
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Drop any remaining rows with NaN
X = X.dropna()
y = y.loc[X.index]

# Save scaler
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, "knn_scaler.pkl"))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize benchmark log ===
benchmark_log = create_benchmark_log()

# === Neighbors to explore ===
neighbors_to_try = [1, 3, 5, 10, 20, 50]

for k in neighbors_to_try:
    print(f"🔍 Training KNN with k={k}")
    start = time.time()

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()
    train_time = round(end - start, 2)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variant = f"k{k}"
    model_filename = os.path.join(models_dir, f"knn_{variant}.pkl")
    joblib.dump(model, model_filename)

    benchmark_log = log_model_result(
        benchmark_log,
        model_name="KNN",
        variant=variant,
        mae=mae,
        r2=r2,
        train_time=train_time,
        params={"n_neighbors": k},
        notes="Distance-based regression (sensitive to scaling)"
    )

# === Save benchmark results ===
save_benchmark_log(benchmark_log, file_name="knn_benchmark_log.csv", log_dir=os.path.join(project_root, "data"))
print("✅ KNN exploration complete.")
