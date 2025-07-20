# xgb_exploration.py
# XGBoost regression: gradient boosting benchmark for fuel price modeling

import pandas as pd
import time
import os
import sys
import joblib
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# === Path setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(os.path.join(project_root, "scripts", "utils"))

from benchmark_utils import create_benchmark_log, log_model_result, save_benchmark_log

# === Paths ===
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# === Load and preprocess ===
df = pd.read_csv(data_path)
y = df['Price']
X = df.drop(columns=['Price'])

# === Scale numeric features ===
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X = X.dropna()
y = y.loc[X.index]

# === Sanitize feature names for XGBoost compatibility ===
import re
def sanitize_column(name):
    name = re.sub(r'\W+', '_', name)
    return re.sub(r'_+', '_', name).strip('_')
X.columns = [sanitize_column(col) for col in X.columns]

# Save scaler
joblib.dump(scaler, os.path.join(models_dir, "xgb_scaler.pkl"))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Benchmark log ===
benchmark_log = create_benchmark_log()

# === Variants to try ===
configs = [
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},     # default-ish
    {"max_depth": 10, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 15, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 20, "learning_rate": 0.01, "n_estimators": 500},
]

# === Train and evaluate ===
for cfg in configs:
    print(f"🚀 Training XGBoost with config: {cfg}")
    start = time.time()

    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()
    train_time = round(end - start, 2)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variant = f"depth{cfg['max_depth']}_lr{cfg['learning_rate']}_n{cfg['n_estimators']}"
    model_filename = os.path.join(models_dir, f"xgb_{variant}.pkl")
    joblib.dump(model, model_filename)

    benchmark_log = log_model_result(
        benchmark_log,
        model_name="XGBoost",
        variant=variant,
        mae=mae,
        r2=r2,
        train_time=train_time,
        params=cfg,
        notes="Boosted trees via XGBoost"
    )

# Save benchmark
save_benchmark_log(benchmark_log, file_name="xgb_benchmark_log.csv", log_dir=os.path.join(project_root, "data"))
print("✅ XGBoost exploration complete.")
