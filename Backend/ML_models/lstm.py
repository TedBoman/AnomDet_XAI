import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from ML_models import model_interface
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


class LSTMModel(model_interface.ModelInterface):

    #Initializes the model
    def __init__(self):
        pass

    #Preprocesses, trains and fits the model
    def run(self, df, time_steps=1):
        features = df.shape[1]
        inputs = Input(shape=(1, features))
        encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)

        decoded = RepeatVector(time_steps)(encoded)
        decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
        outputs = TimeDistributed(Dense(features))(decoded)

        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        self.model = autoencoder

        self.scaler = MinMaxScaler()
        data_normalized = self.scaler.fit_transform(df)
        X = self.__create_sequences(data_normalized, time_steps)

        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]

        self.model.fit(
            X_train, X_train, 
            epochs=25,
            batch_size=256,
            validation_split=0.2,
            verbose=1
        )
    
        reconstructed = self.model.predict(X_test)
        reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
        self.threshold = np.percentile(reconstruction_error, 95)

    #Creates sequences
    def __create_sequences(self, data, time_steps):
        sequences = []
        for i in range(len(data) - time_steps + 1):
            seq = data[i:i + time_steps]
            sequences.append(seq)
        return np.array(sequences)
        
    # Detects anomalies and returns a list of boolean values
    def detect(self, detection_data): # Renamed input variable for clarity
        # Check if the input is an OmniXAI Timeseries object and convert it
        if hasattr(detection_data, 'to_pd') and callable(detection_data.to_pd):
            detection_df = detection_data.to_pd()
        else:
            # Assume it might be called directly with a DataFrame
            detection_df = detection_data

        # --- IMPORTANT FIX from previous analysis ---
        # Use transform, NOT fit_transform, on detection data
        data_normalized = self.scaler.transform(detection_df)
        # --- END FIX ---

        X = self.__create_sequences(data_normalized, 1) # Hardcoded time_steps=1 aligns with model
        reconstructed = self.model.predict(X)
        reconstruction_error = np.mean(np.square(X - reconstructed), axis=(1, 2))
        anomalies = reconstruction_error > self.threshold

        # OmniXAI expects numpy arrays as output
        return np.array(anomalies)
