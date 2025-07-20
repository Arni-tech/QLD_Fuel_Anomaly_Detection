# ridge_exploration.py
# Ridge Regression deep dive: tuning alpha, evaluation, and benchmarking

import pandas as pd
import time
import os
import sys
from sklearn.linear_model import Ridge
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
benchmark_file = os.path.join(project_root, "data", "ridge_benchmark_log.csv")

# === Load and split data ===
df = pd.read_csv(data_path)
y = df['Price']
X = df.drop(columns=['Price'])  # TransactionDateutc already dropped

# === Scale numeric features ===
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Drop any rows with NaNs
X = X.dropna()
y = y.loc[X.index]


# Save the scaler for Ridge (optional, since it's shared)
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, "ridge_scaler.pkl"))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize benchmark log ===
benchmark_log = create_benchmark_log()

# === Alpha values to try ===
alphas = [0.01, 0.1, 1, 10, 100, 500]

for alpha in alphas:
    print(f"🔍 Training Ridge Regression with alpha={alpha}")
    start = time.time()

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()
    train_time = round(end - start, 2)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variant = f"alpha{alpha}"
    model_filename = os.path.join(models_dir, f"ridge_{variant}.pkl")
    joblib.dump(model, model_filename)

    benchmark_log = log_model_result(
        benchmark_log,
        model_name="Ridge",
        variant=variant,
        mae=mae,
        r2=r2,
        train_time=train_time,
        params={"alpha": alpha},
        notes="L2 regularized linear regression"
    )

# === Save benchmark results ===
save_benchmark_log(benchmark_log, file_name="ridge_benchmark_log.csv", log_dir=os.path.join(project_root, "data"))
print("✅ Ridge Regression exploration complete.")
