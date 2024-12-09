import keras
import numpy as np
import pandas as pd
import model_interface
from sklearn.preprocessing import StandardScaler


class LSTMModel(model_interface.ModelInterface):

    def __init__(self):
        self.model = keras.Sequential()

    def run(self, df, time_steps=2880):

        train_size = int(len(df) * .95)
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

        scaler = StandardScaler()
        scaler = scaler.fit(train.iloc[:, 1:])

        train.iloc[:, 1:] = scaler.transform(train.iloc[:, 1:])
        test.iloc[:, 1:] = scaler.transform(test.iloc[:, 1:])

        X_train, y_train = self.__create_dataset(train.iloc[:, 1:], train.iloc[:, 0], time_steps)
        X_test, y_test = self.__create_dataset(test.iloc[:, 1:], test.iloc[:, 0], time_steps)


        self.model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(keras.layers.Dropout(rate=0.2))
        self.model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        self.model.add(keras.layers.LSTM(units=64, return_sequences=True))
        self.model.add(keras.layers.Dropout(rate=0.2))  # Added layer
        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))

        self.model.compile(loss='mae', optimizer='adam')

        self.model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=30,
            validation_split=0.1,
            shuffle=False  # not shuffle time series data because it is history dependant!!!!
        )

        self.detect(X_test)
        return

    def __create_dataset(self, X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i: (i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])

        return np.array(Xs), np.array(ys)

    def detect(self, detection_df):

        X_pred = self.model.predict(detection_df)
        mae_loss = np.mean(np.abs(X_pred - detection_df), axis=1)
        threshold = np.mean(mae_loss) + 3 * np.std(mae_loss)
        boolean_anomalies = mae_loss > threshold

        return boolean_anomalies




