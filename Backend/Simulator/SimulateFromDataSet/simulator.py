# simulator.py

import sys
import time as t
from pathlib import Path
import numpy as np
import pandas as pd

from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
import DBAPI.utils as ut
import FileFormats.read_csv as rcsv

class Simulator:
    """
    Simulates streaming data from a file to a database, with optional anomaly injection.

    Attributes:
        file_path (str): Path to the data file.
        file_extention (str): Extension of the data file.
        x_speedup (int, optional): Speedup factor for the simulation (default: 1).
        chunksize (int, optional): Chunk size for reading the file (default: 1).
        start_time (pd.Timestamp): Start time for the simulation.
    """

    def __init__(self, file_path, file_extention, start_time, x_speedup=1, chunksize=1):
        """
        Initializes Simulator with file path, extension, start time, speedup, and chunk size.
        """
        self.file_path = file_path
        self.file_extention = file_extention
        self.x_speedup = x_speedup
        self.chunksize = chunksize
        self.start_time = start_time

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
        Creates a table in the timeseries database. If the table already exists, 
        it creates a new table with a numbered suffix.

        Args: 
            conn_params: The parameters needed to connect to the database.
            tb_name: The base name of the table.
            columns: The columns for the table.

        Returns:
            str: The actual name of the table created (might include a suffix).
        """
        db_instance = self.init_db(conn_params)
        if not db_instance:
            return None
        
        try:
            db_instance.create_table(tb_name, columns)
            return tb_name  # Return the original name if successful
        except Exception as e:
            db_instance.conn.rollback()
            if "already exists" in str(e):
                i = 1
                new_table_name = f"{tb_name}_{i}"
                while True:
                    try:
                        db_instance.create_table(new_table_name, columns)
                        return new_table_name  # Return the new table name
                    except Exception as ex:
                        db_instance.conn.rollback()
                        if "already exists" in str(ex):
                            i += 1
                            new_table_name = f"{tb_name}_{i}"
                        else:
                            raise ex
            else:
                raise e

    def process_row(self, conn_params, table_name, row, anomaly_settings=None):
        """
        Processes a single row of data, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters
            table_name (str): Name of the table to insert data into
            row (pd.Series): A single row of data to be inserted
            anomaly_settings (list, optional): List of anomaly settings to apply
        """
        # Create a DataFrame from the single row
        df = pd.DataFrame([row])

        # Create a new column to track anomalies
        df['injected_anomaly'] = False
        
        print(anomaly_settings)

        injector = TimeSeriesAnomalyInjector()

        # Inject anomalies if settings are provided
        if anomaly_settings:
            for setting in anomaly_settings:
                print(setting)
                if setting.timestamp:
                    # Check if this row falls within the anomaly time range
                    row_timestamp = row['timestamp']

                    anomaly_start = setting.timestamp
                    anomaly_end = anomaly_start + ut.parse_duration(setting.duration)
                    
                    print(anomaly_start)
                    print(anomaly_end)
                    print(row_timestamp)
                    print(df)
                    sys.stdout.flush()

                    # Check if row timestamp is within anomaly time range
                    if anomaly_start <= row_timestamp <= anomaly_end:
                        print(f"Injecting anomaly on {row_timestamp}")
                        df = injector.inject_anomaly(df, setting)
                        print(df)
                        sys.stdout.flush()
        
        # Insert the row (modified or not) into the database
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='s') 

        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, df)
        print("Inserted row.")
        sys.stdout.flush()

    def start_simulation(self, conn_params, anomaly_settings=None):
        """
        Reads the data file, preprocesses anomaly settings, and inserts data
        into the database row by row, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters
            anomaly_settings (list, optional): List of anomaly settings to apply
        """
        print("Stream job has been started!")
        # Get column names from the first row of the CSV file
        with open(self.file_path, 'r') as f:
            columns = f.readline().strip().split(',')
        
        table_name = self.create_table(conn_params, Path(self.file_path).stem, columns)

        full_df = self.read_file()
        if full_df == None:
            print(f"Fileformat {self.file_extention} not supported!")
            print("Canceling job")
            return

        # Preprocess anomaly settings to convert timestamps to absolute times
        if anomaly_settings:
            for setting in anomaly_settings:
                # Convert timestamp to absolute time if it's not already
                if not isinstance(setting.timestamp, pd.Timestamp):
                    setting.timestamp = self.start_time + pd.to_timedelta(setting.timestamp, unit='s').astype(np.int64) / 1e9
                
                if setting.columns:
                    setting.data_range = []
                    setting.mean = []
                    for col in setting.columns:
                        # Calculate and store the data range of this column
                        data_range = full_df[col].max() - full_df[col].min()
                        setting.data_range.append(data_range)

                        # Calculate and store the mean of this column
                        mean = full_df[col].mean()
                        setting.mean.append(mean)

        time_between_input = full_df.iloc[:, 0].diff().mean()
        print(f"Speedup: {self.x_speedup}")
        print(f"Time between inputs: {time_between_input} seconds")

        # Convert the first column (assume it's timestamps)
        try:
            # Try converting the first column to datetime objects directly
            full_df[full_df.columns[0]] = pd.to_datetime(full_df[full_df.columns[0]])
        except ValueError:
            # If direct conversion fails, try converting to numeric first
            try:
                full_df[full_df.columns[0]] = pd.to_numeric(full_df[full_df.columns[0]])
                full_df[full_df.columns[0]] = pd.to_datetime(full_df[full_df.columns[0]], unit='s')  # Assuming seconds if numeric
            except ValueError:
                print("Error: Could not convert the first column to datetime. Please ensure it's in a valid format.")
                return

        # Calculate time difference in seconds
        time_between_input = full_df.iloc[:, 0].diff().dt.total_seconds().mean()

        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        print(f"Simulation speed between inputs: {time_between_input / self.x_speedup}")
        print("Starting to insert!")

        for index, row in full_df.iterrows():
            print(f"Inserting row {index + 1}")

            # Process the row with potential anomaly injection
            self.process_row(conn_params, table_name, row, anomaly_settings)

            # Sleep between rows, adjusted by speedup
            t.sleep(time_between_input / self.x_speedup)

        print("Inserting done!")
        sys.stdout.flush()

    def read_file(self):
        """
        Reads the data file based on its extension.

        Returns:
            pd.DataFrame: DataFrame containing the data from the file, or None 
                         if the file format is not supported.
        """
        match self.file_extention:
            case '.csv':
                # File is a CSV file. Return a dataframe containing it.
                return rcsv.filetype_csv(self.file_path)
            # Add more fileformats here
            case _:
                # Fileformat not supported
                return None