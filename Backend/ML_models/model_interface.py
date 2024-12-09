from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, df, seq_len):
         pass


    @abstractmethod
    def detect(self, detection_df):
         pass