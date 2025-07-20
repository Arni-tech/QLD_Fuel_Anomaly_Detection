# scripts/model_evaluation/anomaly_detection.py
import os
import joblib
import numpy as np
import pandas as pd

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
model_path = os.path.join(project_root, "models", "rf_n200_dNone.pkl")
scaler_path = os.path.join(project_root, "models", "rf_scaler.pkl")
output_path = os.path.join(project_root, "data", "anomalies.csv")

# Load data, model, scaler
df = pd.read_csv(data_path)
y = df["Price"]
X = df.drop(columns=["Price"])

from sklearn.preprocessing import StandardScaler
numeric_cols = ['Site_Latitude', 'Site_Longitude', 'DayOfWeek', 'Month']
scaler = joblib.load(scaler_path)
X[numeric_cols] = scaler.transform(X[numeric_cols])
X = X.dropna()
y = y.loc[X.index]

# Predict
model = joblib.load(model_path)
y_pred = model.predict(X)
residuals = y - y_pred

# Attach back to DataFrame
df = df.loc[X.index].copy()
df["Prediction"] = y_pred
df["Residual"] = residuals
df["Abs_Residual"] = np.abs(residuals)

# Choose anomaly threshold: e.g. 3 standard deviations
std = residuals.std()
threshold = 3 * std
print(f"Residual Std Dev: {std:.2f} | Using threshold: {threshold:.2f}")

anomalies = df[df["Abs_Residual"] > threshold].sort_values("Abs_Residual", ascending=False)

# Output
print(f"Found {len(anomalies)} anomalies out of {len(df)} records ({100*len(anomalies)/len(df):.3f}%)")
print(anomalies[["SiteId", "Prediction", "Price", "Residual"]].head(10))

anomalies.to_csv(output_path, index=False)
print(f"Anomaly table written to {output_path}")
