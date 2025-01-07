import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras import Sequential
from keras import layers
import numpy as np
import pandas as pd
from ML_models import model_interface
from sklearn.preprocessing import StandardScaler


class LSTMModel(model_interface.ModelInterface):

    #Initializes the model
    def __init__(self):
        self.model = Sequential()

    #Preprocesses, trains and fits the model
    def run(self, df, time_steps=1):
        train_size = len(df)
        self.time_steps = time_steps

        try:
            train = df.iloc[0:train_size - 1]

            scaler = StandardScaler()
            scaler = scaler.fit(train.iloc[:, 1:])

            train.iloc[:, 1:] = scaler.transform(train.iloc[:, 1:])

            X_train, y_train = self.__create_dataset(train.iloc[:, 1:], train.iloc[:, 0])

            self.model.add(layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
            self.model.add(layers.Dropout(rate=0.2))
            self.model.add(layers.RepeatVector(n=X_train.shape[1]))
            self.model.add(layers.LSTM(units=64, return_sequences=True))
            self.model.add(layers.Dropout(rate=0.2))
            self.model.add(layers.TimeDistributed(keras.layers.Dense(1)))


            self.model.compile(loss='mae', optimizer='adam')

            self.model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=time_steps,
                validation_split=0.1,
                shuffle=False  # not shuffle time series data because it is history dependant!!!!
            )
        except Exception as e:
            print(f'ERROR: {e}')

    #Creates the X_train and y_train datasets
    def __create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i: (i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])

        return np.array(Xs), np.array(ys)

    #Detects anomalies and returns a list of boolean values that can be mapped to the original dataset
    def detect(self, detection_df):
        try:
            X_test, _ = self.__create_dataset(detection_df.iloc[:, 1:], detection_df.iloc[:, 0])

            X_pred = self.model.predict(X_test)
            mae_loss = np.mean(np.abs(X_pred - X_test), axis=1)
            threshold = np.mean(mae_loss) + 3 * np.std(mae_loss)
            boolean_anomalies = mae_loss > threshold

            boolean_anomalies = boolean_anomalies[:,0]
            for i in range(self.time_steps):
                boolean_anomalies = np.append(boolean_anomalies, False)
        except Exception as e:
            print(f'ERROR: {e}')

        return boolean_anomalies