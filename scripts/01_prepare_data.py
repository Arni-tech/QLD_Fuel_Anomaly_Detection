# 01_prepare_data.py
# Preprocess and encode raw fuel price data

import pandas as pd
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
raw_path = os.path.join(project_root, "data", "qld_fuel.csv")
processed_path = os.path.join(project_root, "data", "processed_fuel_data.csv")

# === Load raw data ===
df = pd.read_csv(raw_path)

# === Filter out extreme outliers ===
df = df[(df['Price'] > 200) & (df['Price'] < 4000)].copy()

# === Convert TransactionDateutc to datetime ===
df['TransactionDateutc'] = pd.to_datetime(df['TransactionDateutc'])
df['DayOfWeek'] = df['TransactionDateutc'].dt.dayofweek
df['Month'] = df['TransactionDateutc'].dt.month

# === Drop non-numeric and non-encoded columns ===
df = df.drop(columns=[
    'Site_Name',
    'Sites_Address_Line_1',
    'Site_Suburb',
    'TransactionDateutc','Site_State'  # already used for feature extraction
], errors='ignore')

# === One-hot encode categorical features ===
df_encoded = pd.get_dummies(df, columns=['Fuel_Type', 'Site_Brand'], drop_first=True)

# === Save processed output ===
os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
df_encoded.to_csv(processed_path, index=False)

print(f"✅ Preprocessed data saved to: {processed_path}")
