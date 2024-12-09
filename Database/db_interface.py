from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class DBInterface(ABC):
    # Constructor that adds all connection parameters needed to connect to the database to the object
    @abstractmethod
    def __init__(self, conn_params: dict):
        pass
    # Creates a hypertable called table_name with column-names columns
    @abstractmethod
    def create_table(self, table_name: str, columns: list[str]):
        pass
    # Inserts data into the table_name table. The data is a pandas DataFrame with matching columns to the table
    @abstractmethod
    def insert_data(self, table_name: str, data: pd.DataFrame):
        pass
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    @abstractmethod
    def read_data(self, table_name: str, time: datetime):
        pass
    # Deletes the table_name table along with all its data
    @abstractmethod
    def drop_table(self, table_name: str):
        pass