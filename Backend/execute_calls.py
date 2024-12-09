#from ML_models.lstm import LSTMModel
#from ML_models.isolation_forest import IsolationForest
#from Database.db_interface import DBInterface
import pandas as pd
import os

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./injection_methods"
DATASET_DIRECTORY = "../Datasets"

def run_batch(model: str, injection_method: str, file_name: str) -> str:

    dataset_folder = "Datasets"
    full_path = DATASET_DIRECTORY + "/" + file_name
    df = pd.read_csv(full_path)

    match model:
        case "lstm":
            #lstm_instance = LSTMModel()
            #detection_df = lstm_instance.run(df)
            #anomalies = lstm_instance.detect(detection_df)
            pass
        
        case "isolation_forest":
            #if_instance = IsolationForest()
            #detection_df = if_instance.run(df)
            #anomalies = if_instance.detect(detection_df)
            pass
        
        case _:
            raise Exception("Model not found")

# Returns a list of models implemented in MODEL_DIRECTORY
def get_models() -> list:
    models = []
    for path in os.listdir(MODEL_DIRECTORY):
        file_path = MODEL_DIRECTORY + "/" + path
        print(file_path)
        if os.path.isfile(file_path):
            model_name = path.split(".")[0]
            models.append(model_name)

    models.remove("model_interface")
    models.remove("__init__")
    models.remove("setup")
    models.remove("LSTM")
    
    return models

def get_injection_methods() -> list:
    injection_methods = ["not implemented"]
    '''
    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)
    '''
    return injection_methods

def get_datasets() -> list:
    datasets = []
    for path in os.listdir(DATASET_DIRECTORY):
        file_path = DATASET_DIRECTORY + "/" + path
        print(file_path)
        if os.path.isfile(file_path):
            dataset = path
            datasets.append(dataset)

    return datasets