import pandas as pd
import model_lstm
import os

from ML_models.model_lstm import LSTMModel


def read_dataset():
    #REPLACE WITH DATABASE COMMUNICATION
    base_path = os.path.dirname(__file__)
    dataset_path = os.path.join(base_path, 'system-1.csv')
    data = pd.read_csv(dataset_path, low_memory=False)
    return data

if __name__ == "__main__":
    data = read_dataset()
    model = LSTMModel()
    anomalies, errors = model.run(data, 10)
