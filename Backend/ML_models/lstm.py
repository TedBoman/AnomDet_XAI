import keras
import numpy as np
import pandas as pd
import model_interface
from sklearn.preprocessing import StandardScaler


class LSTMModel(model_interface.ModelInterface):

    def run(self, df, seq_len):
        df = pd.read_csv('system-1.csv', parse_dates=['timestamp'], index_col='timestamp')

        train_size = int(len(df) * .95)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
        print(train.shape, test.shape)

        scaler = StandardScaler()
        scaler = scaler.fit(train.iloc[:, 1:])

        train.iloc[:, 1:] = scaler.transform(train.iloc[:, 1:])
        test.iloc[:, 1:] = scaler.transform(test.iloc[:, 1:])


        TIME_STEPS = 30

        X_train, y_train = create_dataset(train.iloc[:, 1:], train.iloc[:, 0], TIME_STEPS)
        X_test, y_test = create_dataset(test.iloc[:, 1:], test.iloc[:, 0], TIME_STEPS)

        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        model.add(keras.layers.LSTM(units=64, return_sequences=True))  # Added layer
        model.add(keras.layers.Dropout(rate=0.2))  # Added layer
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
        # compile using mean absolute error
        model.compile(loss='mae', optimizer='adam')

        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=30,
            validation_split=0.1,
            shuffle=False  # not shuffle time series data because it is history dependant!!!!
        )

        self.detect(X_test, model)
        return

    def detect(self, detection_df, model):

        X_train_pred = model.predict(detection_df)
        train_mae_loss = np.mean(np.abs(X_train_pred - detection_df), axis=1)
        X_test_pred = model.predict(detection_df)
        test_mae_loss_new = np.mean(np.abs(X_test_pred - detection_df), axis=1)
        threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)
        boolean_anomalies = test_mae_loss_new > threshold
        anomalous_indices = np.where(boolean_anomalies)[0]
        anomalous_samples = detection_df[anomalous_indices]
        return anomalous_samples


def create_dataset(X, y, time_steps=2880):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)

