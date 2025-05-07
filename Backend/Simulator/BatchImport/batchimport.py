# batchimport.py
import sys
import psycopg2
import time as t
import multiprocessing as mp
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

#from Simulator.DBAPI.db_interface import DBInterface as db
from timescaledb_api import TimescaleDBAPI as db
from Simulator.AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
import Simulator.DBAPI.utils as ut
from Simulator.DBAPI.debug_utils import DebugLogger as dl
from Simulator.FileFormats.read_csv import read_csv
from Simulator.FileFormats.read_json import read_json

class BatchImporter:
    """
    Imports data from a file into a database in batches, with optional anomaly injection.

    Attributes:
        file_path (str): Path to the data file.
        file_extention (str): Extension of the data file.
        start_time (pd.Timestamp): Start time for the data.
        chunksize (int, optional): Size of each batch (default: 100).
    """

    def __init__(self, file_path, file_extention, start_time, chunksize=100):
        """
        Initializes BatchImporter with file path, extension, start time, and chunk size.
        """
        self.file_path = file_path
        self.chunksize = chunksize
        self.start_time = start_time
        self.file_extention = file_extention

    def init_db(self, conn_params) -> db:
        """
        Returns an instance of the database interface API.

        Args:
            conn_params: A dictionary containing the parameters needed
                         to connect to the timeseries database.
        """
        retry = 0

        while retry < 5:
            db_instance = db(conn_params)
            if db_instance:
                return db_instance
            else:
                time = 3
                while time > 0:
                    dl.debug_print("Retrying in: {time}s")
                    t.sleep(1)
        return None

    def create_table(self, conn_params, tb_name, columns):
        """
        Ensures a table exists in the timeseries database using the DatabaseAPI.

        Args:
            tb_name: The desired base name of the table.
            columns: The columns for the table.

        Returns:
            str: The actual name of the table if newly created.
            int: 1 if the table already exists.
            None: If an error occurred during the process or DB connection failed.
        """
        db_instance = self.init_db(conn_params)
        if not db_instance:
            return None

        result = db_instance.create_table(tb_name, list(columns))

        if result == None:
            # Table already exists, let the calling function know
            return None
        elif isinstance(result, str):
            # Table was created successfully, return the name
            return result
        else: # result is None (error)
            print(f"Error reported by API during creation of table '{tb_name}'.")
            return None

    def process_chunk(self, conn_params, table_name, chunk):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
            table_name: The name of the table.
            chunk (pd.DataFrame): A chunk of data to be inserted.
        """
        # Insert the row (modified or not) into the database
        # Handle timestamp conversion safely
        if 'timestamp' in chunk.columns and not pd.api.types.is_datetime64_any_dtype(chunk['timestamp']):
            dl.debug_print(f"Process_chunk: Timestamp column in table {table_name} is not datetime. Attempting conversion.")
            # Make a copy to avoid SettingWithCopyWarning if chunk is a view
            timestamp_series = chunk['timestamp'].copy() 
            try:
                # First, try standard conversion (handles various string formats, including ISO)
                converted_timestamps = pd.to_datetime(timestamp_series, errors='coerce')

                # If standard conversion results in all NaT, and original wasn't all NaT,
                # it might indicate numeric timestamps (e.g., Unix epochs in seconds).
                if converted_timestamps.isna().all() and not timestamp_series.isna().all():
                    dl.debug_print(f"Process_chunk: Standard to_datetime resulted in all NaT for non-NaT input for table {table_name}. Trying with unit='s'.")
                    # This attempts to interpret numbers as seconds since epoch
                    converted_timestamps = pd.to_datetime(timestamp_series, unit='s', errors='coerce')
                
                chunk.loc[:, 'timestamp'] = converted_timestamps
                
                # Optionally, handle rows where timestamp is still NaT (unconvertible)
                # For example, by dropping them or logging them:
                if chunk['timestamp'].isna().any():
                    dl.debug_print(f"Process_chunk: {chunk['timestamp'].isna().sum()} rows in table {table_name} had unconvertible timestamps.")
                    # chunk = chunk.dropna(subset=['timestamp']) # Uncomment to drop

            except Exception as e:
                dl.print_exception(f"Process_chunk: Critical error converting timestamp in table {table_name}: {e}. Coercing to NaT.")
                chunk.loc[:, 'timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce') # Fallback to coerce

        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, chunk)

    def inject_anomalies_into_chunk(self, chunk, anomaly_settings):
        """
        Injects anomalies into a chunk of data.

        Args:
            chunk (pd.DataFrame): The chunk of data to inject anomalies into.
            anomaly_settings (list): List of anomaly settings.

        Returns:
            pd.DataFrame: The modified chunk with injected anomalies.
        """
        try:
            injector = TimeSeriesAnomalyInjector()
            chunk_start_time = pd.to_datetime(chunk['timestamp'].min(), unit='s')
            chunk_end_time = pd.to_datetime(chunk['timestamp'].max(), unit='s')
            
            for setting in anomaly_settings:
                anomaly_start = setting.timestamp
                anomaly_end = anomaly_start + pd.Timedelta(seconds=ut.parse_duration(setting.duration).total_seconds())

                # Check if the chunk overlaps with the anomaly's time range
                if (chunk_start_time <= anomaly_end) and (chunk_end_time >= anomaly_start):
                    # Inject anomalies
                    dl.debug_print("Anomaly within chunk!")
                    
                    chunk = injector.inject_anomaly(chunk, setting)
                    dl.debug_print(chunk)
                
            return chunk

        except Exception as e:
            dl.print_exception(f"Error injecting anomalies into chunk: {e}")
            return chunk

    def start_simulation(self, conn_params, anomaly_settings=None, table_name=None, timestamp_col_name=None, label_col_name=None):
        """
        Starts the batch data import process.

        Reads the data file, preprocesses anomaly settings, and inserts data
        into the database in chunks, with optional anomaly injection.

        Args:
            conn_params: Database connection parameters.
            anomaly_settings (list, optional): List of anomaly settings to apply.
        """
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)

        dl.debug_print(self.file_path)
        dl.debug_print(self.chunksize)
        dl.debug_print(self.start_time)
        dl.debug_print("Starting to insert!")

        # Preprocess anomaly settings to convert timestamps to absolute times
        if anomaly_settings:
            for setting in anomaly_settings:
                # Convert timestamp to absolute time if it's not already
                if not isinstance(setting.timestamp, pd.Timestamp):
                    setting.timestamp = self.start_time + pd.to_timedelta(setting.timestamp, unit='s')

        # Create a list to store results from async processes
        results = []

        full_df = self.read_file()
        if full_df is None or full_df.empty:
            dl.print_exception(f"Fileformat {self.file_extention} not supported!")
            dl.print_exception("Canceling job")
            return
        
        # --- Drop Unnamed Columns ---
        # Select columns that do not start with 'Unnamed:'
        full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed:')]
        dl.debug_print("Dropped unnamed columns.")
        # --- End Drop Unnamed Columns ---
        
        #full_df = full_df.rename(columns={timestamp_col_name: 'timestamp'}) # Uncomment when passing timestamp column
        full_df.columns.values[0] = "timestamp"

        print(f"renaming columns '{label_col_name}'")
        rename_map = {}
        if label_col_name != 'label':
            rename_map[label_col_name] = 'label'

        # --- Perform Renaming Operation ---
        if rename_map: # Only rename if there's anything to rename
            full_df = full_df.rename(columns=rename_map)
            dl.debug_print("Label column renaming applied.")
            
        # --- Label Column Conversion to 0/1 ---
        if 'label' in full_df.columns:
            dl.debug_print(f"Processing 'label' column. Initial unique values (up to 10): {full_df['label'].dropna().unique()[:10]}")

            def convert_label_value(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower == 'true':
                        return 1
                    if val_lower == 'false':
                        return 0
                elif isinstance(val, bool):
                    return 1 if val else 0
                
                # Check for numeric 1 or 0 (handles int and float)
                # Using np.isclose for float comparison is safer if precision issues were a concern,
                # but direct equality works for exact 0.0 and 1.0.
                if val == 1 or val == 1.0:
                    return 1
                if val == 0 or val == 0.0:
                    return 0
                
                # Default for anything else (other numbers, unhandled strings, None, NaN)
                return 0

            full_df['label'] = full_df['label'].apply(convert_label_value).astype(int)
            
            dl.debug_print(f"Converted 'label' column to 0/1 integers. Unique values after conversion: {full_df['label'].unique()}")
        else:
            dl.debug_print("'label' column not found or specified. Skipping label conversion.")

        
        columns = list(full_df.columns.values)
        
        table_name = self.create_table(conn_params, Path(self.file_path).stem if table_name is None else table_name, columns)
        if table_name == None:
            return 1

        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        dl.debug_print(full_df.head())  # Inspect the parsed DataFrame

        # Set the chunksize to the number of rows in the file / cpu cores available
        self.chunksize = len(full_df.index) / num_processes

        # Create a new column to track anomalies
        full_df['injected_anomaly'] = False
        full_df['is_anomaly'] = False

        # Process chunks with anomaly injection
        for chunk in [full_df[i:i+int(self.chunksize)] for i in range(0, int(len(full_df)), int(self.chunksize))]:
            if anomaly_settings:
                chunk = chunk.copy()

                # Inject anomalies across chunk boundaries
                chunk = self.inject_anomalies_into_chunk(chunk, anomaly_settings)

            # Use apply_async and collect results
            result = pool.apply_async(
                self.process_chunk,
                args=(conn_params, table_name, chunk),
            )
            results.append(result)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        # Optionally, check for any exceptions in the results
        for result in results:
            result.get()  # This will raise any exceptions that occurred in the process

        dl.debug_print("Inserting done!")
        return 1
        

    def read_file(self):
        """
        Reads the data file based on its extension.

        Returns:
            pd.DataFrame: DataFrame containing the data from the file, or None 
                         if the file format is not supported.
        """
        try:
            match self.file_extention:
                case '.csv':
                    # File is a CSV file. Return a dataframe containing it.
                    csv = read_csv(self.file_path)
                    full_df = csv.filetype_csv()
                    return full_df
                case '.json':
                    json = read_json(self.file_path)
                    full_df = json.filetype_json()
                    return full_df
                # Add more fileformats here
                case _:
                    # Fileformat not supported
                    return None
        except Exception as e:
            print(f"Error: {e}")
