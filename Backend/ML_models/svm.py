import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ML_models import model_interface
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input


class SVMModel(model_interface.ModelInterface):
    
    #Initializes the model
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.scaler = StandardScaler()
        
    #Preprocesses, trains and fits the model
    def run(self, df):
        X = df.iloc[:, 1:]
        X_train = X

        X_train = self.scaler.fit_transform(X_train)
        
        train_encoded_data = self.__run_autoencoder(X_train)
        self.model.fit(train_encoded_data)


    #Runs the autoencoder to find "normal" data
    def __run_autoencoder(self, X_train):
        input_dim = X_train.shape[1] 
        encoding_dim = 10  
        
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded) 

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1)

        encoder = Model(inputs=input_layer, outputs=encoded)
        encoded_data = encoder.predict(X_train)
        return encoded_data

    #Detects anomalies and returns a list of boolean values that can be mapped to the original dataset
    def detect(self, df):
        X_test = df.iloc[:, 1:]
        X_test = self.scaler.fit_transform(X_test)
        test_encoded_data = self.__run_autoencoder(X_test)

        decision_function = self.model.decision_function(test_encoded_data)
        threshold = np.percentile(decision_function, 5)
        
        adjusted_predictions = (decision_function < threshold).astype(int)
        boolean_anomalies = adjusted_predictions == 1
        return boolean_anomalies
                