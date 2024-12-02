from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def preprocess(self, df, seq_len):
        pass

    def split(self, X):
        pass

    def model_train(self, X_train, model):
        pass

    def model_build(self, input_shape):
        pass

    def model_evaluate(self, X_test, model):
        pass

    def detect(self, *args, **kwargs):
        pass

    def model_plot(self, anomalies):
        pass
