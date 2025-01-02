import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ML_models import model_interface
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from keras.models import Model, Sequential
from keras.layers import Dense, Input


class SVM(model_interface.ModelInterface):
    
    #Initializes the model
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        
    #Preprocesses, trains and fits the model
    def run(self, df):
        train_size = int(len(df) * 0.5)
        
        X = df.iloc[:, 1:]
        X_train = X.iloc[:train_size]
        train_encoded_data = self.__run_autoencoder(X_train)
        self.model.fit(train_encoded_data)

        test_encoded_data = self.__run_autoencoder(X)
        self.detect(test_encoded_data)

    #Runs the autoencoder to find "normal" data
    def __run_autoencoder(self, X_train):
        input_dim = X_train.shape[1] 
        encoding_dim = 10  
        
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded) 

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

        encoder = Model(inputs=input_layer, outputs=encoded)
        encoded_data = encoder.predict(X_train)
        return encoded_data

    #Detects anomalies and returns a list of boolean values that can be mapped to the original dataset
    def detect(self, df):
        self.model.predict(df)
        
        decision_function = self.model.decision_function(df)
        threshold = np.percentile(decision_function, 5)
        adjusted_predictions = (decision_function < threshold).astype(int)
        
        boolean_anomalies = adjusted_predictions == 0
        return boolean_anomalies
                