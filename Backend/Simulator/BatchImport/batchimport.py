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

        i = 1
        new_table_name = tb_name
        while True:
            try:
                db_instance.create_table(new_table_name, columns)
                return new_table_name
            except psycopg2.errors.DuplicateTable:  # Catch the specific exception
                i += 1
                new_table_name = f"{tb_name}_{i}"
            except (psycopg2.errors.OperationalError, 
                    psycopg2.errors.ProgrammingError) as e:
                # Handle or log other database-related errors
                dl.print_exception(f"Database error creating table: {e}")
                raise  # Or re-raise if you want to stop execution
            except Exception as e:  # Catch other unexpected errors
                dl.print_exception(f"Unexpected error creating table: {e}")
                raise

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
        if isinstance(chunk['timestamp'].iloc[0], pd.Timestamp):
            # Already a timestamp, no conversion needed
            pass
        else:
            try:
                # First try direct conversion from Unix timestamp
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s')
            except (ValueError, OutOfBoundsDatetime):
                # If that fails, try converting through datetime
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        db_instance = self.init_db(conn_params)
        db_instance.insert_data_no_helper(table_name, chunk)

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

            # Create a new column to track anomalies
            chunk['injected_anomaly'] = False
            
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

    def start_simulation(self, conn_params, anomaly_settings=None, table_name=None):
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

        with open(self.file_path, 'r') as f:
            columns = f.readline().strip().split(',')

        table_name = self.create_table(conn_params, Path(self.file_path).stem if table_name is None else table_name, columns)

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
            
        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        dl.debug_print(full_df.head())  # Inspect the parsed DataFrame

        # Set the chunksize to the number of rows in the file / cpu cores available
        self.chunksize = len(full_df.index) / num_processes

        # Process chunks with anomaly injection
        for chunk in [full_df[i:i+int(self.chunksize)] for i in range(0, int(len(full_df)), int(self.chunksize))]:
            if anomaly_settings:
                chunk = chunk.copy()
                # If timestamps need adjustment, add start_time explicitly
                chunk.loc[:, chunk.columns[0]] = chunk.loc[:, chunk.columns[0]].apply(
                    lambda x: self.start_time + pd.Timedelta(seconds=x.timestamp())
                    if isinstance(x, datetime.datetime) and x < self.start_time
                    else x
                )

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
                csv = read_csv(self.file_path)
                full_df = csv.filetype_csv()
                return full_df
            # Add more fileformats here
            case _:
                # Fileformat not supported
                return None
            
