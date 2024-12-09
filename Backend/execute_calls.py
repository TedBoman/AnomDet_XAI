from Backend.ML_models.lstm import LSTMModel
from Backend.ML_models.isolation_forest import IsolationForest
from Database.db_interface import DBInterface
import pandas as pd
import os

def run_batch(model: str, injection_method: str, file_path: str) -> str:

    dataset_folder = "Datasets"
    full_path = os.path.join(dataset_folder, file_path)
    df = pd.read_csv(full_path)

    match model:
        case "lstm":
            lstm_instance = LSTMModel()
            detection_df = lstm_instance.run(df)
            anomalies = lstm_instance.detect(detection_df)
            pass
        
        case "isolation forest":
            if_instance = IsolationForest()
            detection_df = if_instance.run(df)
            anomalies = if_instance.detect(detection_df)
            pass

