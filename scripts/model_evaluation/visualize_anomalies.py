# visualize_anomalies.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
anomalies_path = os.path.join(project_root, "data", "anomalies.csv")
full_data_path = os.path.join(project_root, "data", "processed_fuel_data.csv")

# Read in anomalies and (optionally) main data
anoms = pd.read_csv(anomalies_path)

# If your anomalies table doesn't have these, merge back with processed data for plotting info
if not {'Site_Latitude', 'Site_Longitude'}.issubset(anoms.columns):
    full_df = pd.read_csv(full_data_path)
    anoms = anoms.merge(full_df[['SiteId', 'Site_Latitude', 'Site_Longitude']], on="SiteId", how="left")

# --- (1) Scatterplot: Anomalies on Map ---
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=anoms,
    x="Site_Longitude",
    y="Site_Latitude",
    hue="Residual",
    palette="coolwarm",
    size=abs(anoms["Residual"]),
    sizes=(20, 200),
    alpha=0.7,
    legend=False
)
plt.title("Anomalies on Map (Colored by Residual)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(project_root, "data", "anomaly_map.png"))
plt.show()

# --- (2) Anomalies over Time (if Timestamp/Date available) ---
if "TransactionDateutc" in anoms.columns:
    anoms["TransactionDateutc"] = pd.to_datetime(anoms["TransactionDateutc"])
    anoms["Month"] = anoms["TransactionDateutc"].dt.to_period("M")
    plt.figure(figsize=(12, 6))
    anoms.groupby("Month").size().plot(kind="bar", color="coral")
    plt.title("Number of Anomalies per Month")
    plt.ylabel("Count")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "data", "anomaly_time_series.png"))
    plt.show()

# --- (3) Bar chart: Anomalies by Fuel Type ---
# If your processed data has one-hot columns, find the max per row
fuel_cols = [col for col in anoms.columns if col.startswith("Fuel_Type_")]
if fuel_cols:
    def fuel_type(row):
        for col in fuel_cols:
            if row[col] == 1 or row[col] == True:
                return col.replace("Fuel_Type_", "")
        return "Unknown"
    anoms["Fuel_Type"] = anoms.apply(fuel_type, axis=1)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=anoms, x="Fuel_Type", order=anoms["Fuel_Type"].value_counts().index)
    plt.title("Anomaly Counts by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Anomaly Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "data", "anomaly_fueltype_bar.png"))
    plt.show()

print("✅ Anomaly visualizations saved to the data/ directory.")
