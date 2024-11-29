# simulator.py

import time
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db

class Simulator:
    def __init__(self, file_path, x_speedup=1, chunksize=1):
        self.file_path = file_path
        self.x_speedup = x_speedup
        self.chunksize = chunksize

    def init_db(self, conn_params) -> db:
        """
        Returns an instance of the database interface API.

        Args:
            conn_params: A dictionary containing the parameters needed
                         to connect to the timeseries database.
        """
        db_instance = db(conn_params)
        return db_instance

    def create_table(self, conn_params, tb_name, columns):
        """
        Creates a table in the timeseries database.

        Args: 
            conn_params: The parameters needed to connect to the database.
            tb_name: The name of the table.
            columns: The columns for the table.
        """
        db_instance = self.init_db(conn_params)
        db_instance.create_table(tb_name, columns)

    def process_entry(self, conn_params, table_name, chunk):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            chunk (pd.DataFrame): A chunk of data to be inserted.
        """
        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, chunk)

    def filetype_csv(self, conn_params):
        """
        Processes a CSV file.
        
        Args:
            conn_params: The parameters needed to connect to the database.
        """
        # Get column names from the first row of the CSV file
        with open(self.file_path, 'r') as f:
            columns = f.readline().strip().split(',')
        
        self.create_table(conn_params, Path(self.file_path).stem, columns)

        dataindex = 1
        time_between_input = self.get_time_diffs_pandas().mean()
        print(f"Time between inputs: {time_between_input} seconds")
        print("Starting to insert!")
        for data in pd.read_csv(self.file_path, chunksize=self.chunksize):
            print(f"Inserting data {dataindex}")
            self.process_entry(conn_params, Path(self.file_path).stem, data)
            dataindex += 1
            time.sleep(time_between_input / self.x_speedup)

        print("Inserting done!")

    def get_time_diffs_pandas(self):
        """Reads the CSV file and calculates time differences between entries in the first column.

        Returns:
            pandas.Series: A pandas Series containing the time differences.
        """
        df = pd.read_csv(self.file_path)
        time_diffs = df.iloc[:, 0].diff()  # Calculate differences between consecutive values in the first column
        return time_diffs