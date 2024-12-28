from ML_models.lstm import LSTMModel
from ML_models.isolation_forest import IsolationForest
import pandas as pd
import numpy as np
import os
from socket import socket

# Third-Party
import threading
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from typing import Union, List, Optional, Dict

# Custom
from Simulator.SimulatorEngine import SimulatorEngine as se

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./Simulator/AnomalyInjector/InjectionMethods"
DATASET_DIRECTORY = "./Datasets"

class AnomalySetting:
    def __init__(self, anomaly_type: str, timestamp: int, magnitude: int, 
                 percentage: int, columns: List[str] = None, duration: str = None,
                 data_range: List[float] = None, mean: List[float] = None):
        self.anomaly_type = anomaly_type
        self.timestamp = timestamp
        self.magnitude = magnitude
        self.percentage = percentage
        self.duration = duration
        self.columns = columns
        self.data_range = data_range
        self.mean = mean

class Job:
    def __init__(self, filepath: str, simulation_type,speedup: int = None, anomaly_settings: List[AnomalySetting] = None, table_name: str = None):
        self.filepath = filepath
        self.anomaly_settings = anomaly_settings
        self.simulation_type = simulation_type
        self.speedup = speedup
        self.table_name = table_name

# Starts processing of dataset in one batch
def run_batch(model: str, path: str, name: str, inj_params: dict=None) -> None:
    print("Starting Batch-job!")
    if inj_params is not None:
        anomaly = AnomalySetting(
        inj_params.get("anomaly_type", None),
        inj_params.get("timestamp", None),
        inj_params.get("magnitude", None),
        inj_params.get("percentage", None),
        inj_params.get("duration", None),
        inj_params.get("columns", None)) 
        batch_job = Job(filepath=path, anomaly_settings=anomaly, simulation_type="batch", speedup=None, table_name=name)
    else:
        batch_job = Job(filepath=path, simulation_type="batch", speedup=None, table_name=name)
    sim_engine = se()
    sim_engine.main(batch_job)

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
    print("Starting Stream-job!")
    if inj_params is not None:
        anomaly = AnomalySetting(
            inj_params.get("anomaly_type", None),
            inj_params.get("timestamp", None),
            inj_params.get("magnitude", None),
            inj_params.get("percentage", None),
            inj_params.get("duration", None),
            inj_params.get("columns", None)
        ) 
        stream_job = Job(filepath=path, anomaly_settings=anomaly, simulation_type="stream", speedup=speedup, table_name=name)
    else:
        stream_job = Job(filepath=path, simulation_type="stream", speedup=speedup, table_name=name)

    sim_engine = se()
    sim_engine.main(stream_job)
    
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
    injection_methods = []

    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)

    injection_methods.remove("__init__")
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

