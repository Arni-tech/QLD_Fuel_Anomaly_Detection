# scripts/model_evaluation/residual_analysis.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# === Paths ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")
model_path = os.path.join(project_root, "models", "rf_n200_dNone.pkl")  # adjust as needed

# === Load Data and Model ===
df = pd.read_csv(data_path)
model = joblib.load(model_path)
    
# --- Reconstruct Fuel_Type from one-hot columns ---
fuel_type_columns = [col for col in df.columns if col.startswith("Fuel_Type_")]
df['Fuel_Type'] = df[fuel_type_columns].idxmax(axis=1).str.replace("Fuel_Type_", "")

# Save for plotting before any further processing
df_plot = df[['Fuel_Type', 'Price']].copy()

# === Prepare Features and Target ===
y = df["Price"]
# Drop any columns not used for prediction
X = df.drop(columns=["Price", "Fuel_Type"])  # Also drop new Fuel_Type col

# Ensure column order matches model input
X = X.loc[:, model.feature_names_in_]

# === Predict and Compute Residuals ===
y_pred = model.predict(X)
residuals = y - y_pred

# === Merge Predictions/Residuals for Plotting ===
df_plot = df_plot.loc[X.index].copy()  # align index
df_plot["Prediction"] = y_pred
df_plot["Residual"] = residuals

# === Plot 1: Residual Distribution ===
plt.figure(figsize=(12, 6))
sns.histplot(df_plot["Residual"], bins=100, kde=True, color="coral")
plt.axvline(0, linestyle="--", color="black")
plt.title("Residual Distribution (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(project_root, "data", "residual_distribution.png"))
plt.close()

# === Plot 2: Residuals by Fuel Type ===
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_plot, x="Fuel_Type", y="Residual")
plt.xticks(rotation=45)
plt.title("Residuals by Fuel Type")
plt.tight_layout()
plt.savefig(os.path.join(project_root, "data", "residuals_by_fuel_type.png"))
plt.close()

# === Plot 3: Predicted vs Actual ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_plot["Price"], y=df_plot["Prediction"], alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.tight_layout()
plt.savefig(os.path.join(project_root, "data", "predicted_vs_actual.png"))
plt.close()

# === Print overall metrics ===
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"📊 Overall MAE: {mae:.2f}, R²: {r2:.2f}")
print("✅ Residual analysis plots saved to /data/")
