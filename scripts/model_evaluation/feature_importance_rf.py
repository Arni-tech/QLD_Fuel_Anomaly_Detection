# scripts/evaluation/feature_importance_rf.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = os.path.join(project_root, "models", "rf_n200_dNone.pkl")
data_dir = os.path.join(project_root, "data")
model = joblib.load(model_path)

features = model.feature_names_in_
importances = model.feature_importances_

df = pd.DataFrame({"Feature": features, "Importance": importances})
df = df.sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Importance", y="Feature", palette="crest")
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance_random_forest.png"), dpi=300)
plt.show()
