import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def model_preprocess(data, sequence_len):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X = []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i + time_steps])

    return np.array(X), scaler

def split_data(X):
    X_train, Xtest = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, Xtest


def model_build(input_shape):
    def build_model(input_shape):
        model = Sequential([
            LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
            LSTM(64, activation='relu', return_sequences=False),
            RepeatVector(input_shape[0]),
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(input_shape[1]))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model


def model_train(X_train, model):
    model.fit(X_train, X_train, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

def model_evaluate(X_test, model):
    reconstructed = model.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructed, 2), axis=(1, 2))
    threshold = np.mean(mse) + 3 * np.std(mse)
    return reconstructed, mse, threshold

def model_detect(mse, threshold):
    return mse > threshold

def model_plot(anomalies):
    number_of_anomalies = np.sum(anomalies)
    print(f"Number of anomalies detected: {number_of_anomalies}")
