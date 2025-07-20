# scripts/evaluation/feature_importance_xgb.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(project_root, "models", "xgb_depth10_lr0.05_n200.pkl")
data_dir = os.path.join(project_root, "data")
model = joblib.load(model_path)

features = model.feature_names_in_
importances = model.feature_importances_

df = pd.DataFrame({"Feature": features, "Importance": importances})
df = df.sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Importance", y="Feature", palette="mako")
plt.title("Top 15 Feature Importances - XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance_xgboost.png"), dpi=300)
plt.show()
