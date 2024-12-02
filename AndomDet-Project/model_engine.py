import interface
import data_read


def run_lstm(model:interface.ModelInterface):
    data_frame = data_read.read_dataset()
    sequence_len = 30
    # Preprocess & split data
    X, scaler = model.preprocess(data_frame, sequence_len)
    X_train, X_test = model.split(X)
    # Build Model
    model_instance = model.model_build(X_train.shape[1:])
    model_instance.summary()
    # Train model
    model.model_train(X_train, model_instance)
    # Evaluate performance
    reconstructed, mse, threshold = model.model_evaluate(X_test, model_instance)
    anomalies = model.detect(mse, threshold)
    model.model_plot(anomalies)
    return





