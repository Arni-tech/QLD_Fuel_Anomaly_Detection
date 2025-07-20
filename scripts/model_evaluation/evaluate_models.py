# scripts/evaluation/evaluate_benchmarks.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(project_root, "data")

benchmark_logs = [
    "rf_benchmark_log.csv",
    "ridge_benchmark_log.csv",
    "knn_benchmark_log.csv",
    "mlp_benchmark_log.csv",
    "lgbm_benchmark_log.csv",
    "xgb_benchmark_log.csv"
]

frames = []
for file in benchmark_logs:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        frames.append(pd.read_csv(path))

df = pd.concat(frames, ignore_index=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="MAE", y="R2", hue="Model", style="Model", s=100)
plt.title("Model Performance: MAE vs R²")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "model_comparison_mae_r2.png"))
plt.show()
