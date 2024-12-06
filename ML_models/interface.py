from abc import ABC, abstractmethod

class ModelInterface(ABC):
     @abstractmethod
     def run(self, df, seq_len):
         pass

#Implemented detect function to work with both single-point data and batch datasets.
     @abstractmethod
     def detect(self, detection_df, seq_len):
         pass

