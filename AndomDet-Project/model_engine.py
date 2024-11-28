import pandas as pd
import os

import model_lstm

def run_model():
    data_frame = read_dataset()
    sequence_len = 30
    #Preprocess & split data
    X, scaler = model_lstm.model_preprocess(data_frame, sequence_len)
    X_train, X_test = model_lstm.split_data(X)
    #Build Model
    model = model_lstm.model_build(X_train.shape[1:])
    model.summary()
    #Train model
    model_lstm.model_train(X_train, model)
    #Evaluate performance
    reconstructed, mse, threshold = model_lstm.model_evaluate(X_test, model)
    anomalies = model_lstm.model_detect(mse, threshold)
    model_lstm.model_plot(anomalies)



    pass

def read_dataset():
    #REPLACE WITH DATABASE COMMUNICATION
    base_path = os.path.dirname(__file__)
    dataset_path = os.path.join(base_path, 'system-1.csv')
    data = pd.read_csv(dataset_path, low_memory=False)
    return data
