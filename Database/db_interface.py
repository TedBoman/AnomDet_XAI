from abc import ABC, abstractmethod
import pandas as pd

class DBInterface(ABC):
    @abstractmethod
    def __init__(self, conn_params: dict):
        pass

    @abstractmethod
    def create_and_insert_table(self, df: pd.Dataframe):
        pass

    @abstractmethod
    def read_data(self, table_name: str, time_step: int):
        pass

    @abstractmethod
    def drop_table(self, table_name: str):
        pass