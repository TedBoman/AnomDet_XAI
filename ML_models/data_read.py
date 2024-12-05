import pandas as pd
import os

def read_dataset():
    #REPLACE WITH DATABASE COMMUNICATION
    base_path = os.path.dirname(__file__)
    dataset_path = os.path.join(base_path, 'system-1.csv')
    data = pd.read_csv(dataset_path, low_memory=False)
    return data