import psycopg2
from psycopg2.extras import execute_values
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import numpy as np

class DBInterface(ABC):
    # Constructor that initiates a connection to the database and adds the connection to self.conn
    def __init__(self, conn_params: dict):
        """
        Initializes a connection to the TimescaleDB database.

        Args:
            conn_params (dict): A dictionary containing the connection parameters.
                                 Example: {"dbname": "your_db", "user": "your_user", ...}
        """

        CONNECTION = f"postgres://{conn_params['user']}:{conn_params['passwd']}@{conn_params.get('host', 'localhost')}:{conn_params.get('port', 5432)}/{conn_params['dbname']}"
        self.conn = psycopg2.connect(CONNECTION)
        self.cursor = self.conn.cursor()

    # Creates a hypertable called table_name with column-names columns
    def create_table(self, table_name: str, columns: list[str]):
        # Adjust types based on your actual data
        columns_with_types = columns.copy()
        for i in range(len(columns)):
            columns[i] = f'\"{columns[i]}\"'
            # Use appropriate types based on your data
            columns_with_types[i] = f'{columns[i]} DOUBLE PRECISION'
        columns_with_types[0] = f'{columns[0]} DOUBLE PRECISION'  
        
        query_create_table = f'CREATE TABLE "{table_name}" ({",".join(columns_with_types)});'
        self.cursor.execute(query_create_table)
        self.conn.commit()

    def insert_data(self, table_name: str, data: pd.DataFrame):
        """
        Inserts data into the specified table.
        """
        with self.conn.cursor() as cur:
            columns = ', '.join([f'"{col}"' for col in data.columns])
            query = f"INSERT INTO \"{table_name}\" ({columns}) VALUES %s"

            try:
                # Convert DataFrame to list of tuples, with type conversion
                values = [tuple(
                    float(x) if isinstance(x, (np.float64, np.float32)) else x 
                    for x in row
                ) for row in data.values]

                execute_values(cur, query, values)
                self.conn.commit()
                print(f"Data insertion successful for table: {table_name}")
            except Exception as e:
                self.conn.rollback()
                print(f"Error inserting data into {table_name}: {e}")

    # Reads data from the table_name table
    def read_data(self, table_name: str, time: datetime):
        pass

    # Deletes the table_name table along with all its data
    def drop_table(self, table_name: str):
        pass