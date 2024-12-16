# simulator.py

import sys
import time as t
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
from DBAPI import utils as ut
import datetime

class Simulator:
    def __init__(self, file_path, start_time, x_speedup=1, chunksize=1):
        self.file_path = file_path
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
                    except Exception as e:
                        db_instance.conn.rollback()
                        if "already exists" in str(e):
                            i += 1
                            new_table_name = f"{tb_name}_{i}"
                        else:
                            raise e
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
        df['timestamp'] = df['timestamp'].astype(np.int64) / 1e9
        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, df)
        print("Inserted row.")
        sys.stdout.flush()

    def filetype_csv(self, conn_params,anomaly_settings=None):
        """
        Processes a CSV file row by row with optional anomaly injection.

        Args:
            conn_params: Database connection parameters
            anomaly_settings (list, optional): List of anomaly settings to apply
        """
        print("Stream job has been started!")
        # Get column names from the first row of the CSV file
        with open(self.file_path, 'r') as f:
            columns = f.readline().strip().split(',')
        
        table_name = self.create_table(conn_params, Path(self.file_path).stem, columns)

        # Preprocess anomaly settings to convert timestamps to absolute times
        if anomaly_settings:
            for setting in anomaly_settings:
                # Convert timestamp to absolute time if it's not already
                if not isinstance(setting.timestamp, pd.Timestamp):
                    setting.timestamp = self.start_time + pd.to_timedelta(setting.timestamp, unit='s').astype(np.int64) / 1e9

        # Read the CSV and process row by row
        full_df = pd.read_csv(self.file_path)
        time_between_input = full_df.iloc[:, 0].diff().mean()
        print(f"Speedup: {self.x_speedup}")
        print(f"Time between inputs: {time_between_input} seconds")

        full_df = pd.read_csv(self.file_path)

        # Convert the first column (assume it's timestamps in seconds)
        full_df[full_df.columns[0]] = self.start_time + pd.to_timedelta(
            full_df[full_df.columns[0]].astype(float), unit='s'
        )

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

    def get_time_diffs_pandas(self):
        """Reads the CSV file and calculates time differences between entries in the first column.

        Returns:
            pandas.Series: A pandas Series containing the time differences.
        """
        df = pd.read_csv(self.file_path)
        time_diffs = df.iloc[:, 0].diff()  # Calculate differences between consecutive values in the first column
        return time_diffs