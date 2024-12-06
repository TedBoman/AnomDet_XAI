import numpy as np
import interface
import tensorflow as tf
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(interface.ModelInterface):

        def run(self, df, seq_len):
            X, scaler = self.__preprocess(df, seq_len)
            print("Input shape: ", X.shape)
            X_train, X_test = self.__split(X)
            self.__build(input_shape=(X_train.shape[0], X_train.shape[1]))
            self.__train(X_train)
            anomalies, errors = self.detect(df, seq_len)

            return anomalies, errors

        def __preprocess(self, df, seq_len):
            scaler = MinMaxScaler(feature_range=(0, 1))
            df = scaler.fit_transform(df)
            '''X = []
            for i in range(len(data_scaled) - seq_len):
                X.append(data_scaled[i:i + seq_len])'''

            return np.array(df), scaler

        def __split(self, X):
            X_train, Xtest = train_test_split(X, test_size=0.2, random_state=42)
            return X_train, Xtest

        def __build(self, input_shape):
            model = models.Sequential([
                layers.LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
                layers.LSTM(64, activation='relu', return_sequences=False),
                layers.RepeatVector(input_shape[0]),
                layers.LSTM(64, activation='relu', return_sequences=True),
                layers.LSTM(128, activation='relu', return_sequences=True),
                layers.TimeDistributed(layers.Dense(input_shape[1]))
            ])
            model.compile(optimizer='adam', loss='mse')
            self.model = model

        def __train(self, X_train):
            self.model.fit(X_train, X_train, epochs=3,
            batch_size=64, validation_split=0.1, verbose=1)

        def detect(self, detection_df, seq_len):
            X, scaler = self.__preprocess(detection_df, seq_len)
            print("Input shape: ", X.shape)
            predicted = self.model.predict(X)
            print("Shape before: ", predicted.shape)
            predicted_rescaled = scaler.inverse_transform(predicted)

            threshold = 0.1
            errors = np.abs(predicted_rescaled - detection_df[seq_len:])
            anomalies = errors > threshold

            return anomalies, errors


