from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class DBInterface(ABC):
    # Constructor that initiates a connection to the database and adds the connection to self.conn
    @abstractmethod
    def __init__(self, conn_params: dict):
        pass
    # Creates a hypertable called table_name with column-names columns
    @abstractmethod
    def create_table(self, table_name: str, columns: list[str]):
        pass
    # Inserts data into the table_name table
    @abstractmethod
    def insert_data(self, table_name: str, data: pd.DataFrame):
        pass
    # Reads data from the table_name table
    @abstractmethod
    def read_data(self, table_name: str, time: datetime):
        pass
    # Deletes the table_name table along with all its data
    @abstractmethod
    def drop_table(self, table_name: str):
        pass