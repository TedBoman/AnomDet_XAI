from ML_models.lstm import LSTMModel
from ML_models.isolation_forest import IsolationForest
import pandas as pd
import numpy as np
import os
from socket import socket

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./injection_methods"
DATASET_DIRECTORY = "./Datasets"

# Starts processing of dataset in one batch
def run_batch(model: str, path: str, name: str, inj_params: dict=None) -> None:

    #Removing the "is_injected" & "is_anomaly" columns
    feature_df = df.iloc[:, :-2]

    #Creating an instance of the model
    match model:
        case "lstm":
            time_steps=30
            lstm_instance = LSTMModel()
            lstm_instance.run(df.iloc[:, :-2], time_steps)
            anomalies = lstm_instance.detect(df.iloc[:, :-2])
            try: 
                df["is_anomaly"] = anomalies
            except Exception as e:
                print(f'ERROR: {e}')
        
        case "isolation_forest":
            if_instance = IsolationForest()
            if_instance.run(df.iloc[:, :-2])

            anomalies = if_instance.detect(df.iloc[:, :-2])
            df["is_anoamaly"] = anomalies
        
        case _:
            raise Exception("Model not found")

# Starts processing of dataset in one batch
def run_stream(model: str, path: str, name: str, speedup: int, inj_params: dict=None) -> None:
    pass

# Returns a list of models implemented in MODEL_DIRECTORY
def get_models() -> list:
    models = []
    for path in os.listdir(MODEL_DIRECTORY):
        file_path = MODEL_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            model_name = path.split(".")[0]
            models.append(model_name)

    # Removing the __init__, setup files and the .env file
    models.remove("")
    models.remove("model_interface")
    models.remove("__init__")
    models.remove("setup")
    
    return models

# Returns a list of injection methods implemented in INJECTION_METHOD_DIRECTORY
def get_injection_methods() -> list:
    injection_methods = ["not implemented"]
    '''
    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)
    '''
    return injection_methods

# Fetching datasets from the dataset directory
def get_datasets() -> list:
    datasets = []
    for path in os.listdir(DATASET_DIRECTORY):
        file_path = DATASET_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            dataset = path
            datasets.append(dataset)

    return datasets

# Get all columns of the table of a running job
def get_columns(name: str, db_api: ) -> list:
    return db_api.get_columns(name)

# Gets content of complete file to the backend
def import_dataset(conn: socket, path: str, timestamp_column: str) -> None:
    file = open(path, "w")
    data = conn.recv(1024).decode("utf-8")
    while data:
        file.write(data)
        data = conn.recv(1024).decode("utf-8")

    file.close()
    
    # Change the timestamp column name to timestamp and move it to the first column
    df = pd.read_csv(path)
    df.rename(columns={timestamp_column: "timestamp"}, inplace=True)
    cols = df.columns.tolist()
    cols.remove("timestamp")
    cols = ["timestamp"] + cols
    df = df[cols]
    df.to_csv(path, index=False)