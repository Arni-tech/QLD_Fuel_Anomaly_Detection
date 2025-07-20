import pandas as pd
from datetime import datetime
import os

def create_benchmark_log():
    return pd.DataFrame(columns=[
        "Model", "Variant", "MAE", "R2", "Train_Time", 
        "Hyperparameters", "Notes"
    ])

def log_model_result(log_df, model_name, variant, mae, r2, train_time, params, notes=""):
    new_row = pd.DataFrame([{
        "Model": model_name,
        "Variant": variant,
        "MAE": mae,
        "R2": r2,
        "Train_Time": train_time,
        "Hyperparameters": str(params),
        "Notes": notes
    }])
    return pd.concat([log_df, new_row], ignore_index=True)

def save_benchmark_log(log_df, file_name="model_benchmark_log.csv", log_dir="../data"):
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, file_name)
    log_df.to_csv(file_path, index=False)
    print(f"📄 Benchmark log saved to: {file_path}")
