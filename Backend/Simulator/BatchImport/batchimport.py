# batchimport.py
import time as t
import multiprocessing as mp
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
import DBAPI.utils as ut
import FileFormats.read_csv as rcsv

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
                    print("Retrying in: {time}s")
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
                return new_table_name  # Return the table name
            except Exception as e:
                db_instance.conn.rollback()
                if "already exists" in str(e):
                    i += 1
                    new_table_name = f"{tb_name}_{i}"
                else:
                    raise e  # Re-raise other exceptions

    def process_chunk(self, conn_params, table_name, chunk):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
            table_name: The name of the table.
            chunk (pd.DataFrame): A chunk of data to be inserted.
        """
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'].astype(np.int64), unit='s')
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
            chunk_start_time = chunk['timestamp'].min()
            chunk_end_time = chunk['timestamp'].max()

            # Create a new column to track anomalies
            chunk['injected_anomaly'] = False
            
            modified_chunk = None # Init the modified chunk

            for setting in anomaly_settings:
                start_time = setting.timestamp
                end_time = start_time + ut.parse_duration(setting.duration)

                # Check if the chunk overlaps with the anomaly's time range
                if (chunk_start_time <= end_time) and (chunk_end_time >= start_time):
                    # Inject anomalies
                    modified_chunk = injector.inject_anomaly(chunk, setting)

            if modified_chunk:
                return modified_chunk
            else:
                return chunk

        except Exception as e:
            print(f"Error injecting anomalies into chunk: {e}")
            return chunk

    def start_simulation(self, conn_params, anomaly_settings=None):
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

        table_name = self.create_table(conn_params, Path(self.file_path).stem, columns)

        print("Starting to insert!")

        # Preprocess anomaly settings to convert timestamps to absolute times
        if anomaly_settings:
            for setting in anomaly_settings:
                # Convert timestamp to absolute time if it's not already
                if not isinstance(setting.timestamp, pd.Timestamp):
                    setting.timestamp = self.start_time + pd.to_timedelta(setting.timestamp, unit='s')

        # Create a list to store results from async processes
        results = []

        full_df = self.read_file()
        if full_df is None:
            print(f"Fileformat {self.file_extention} not supported!")
            print("Canceling job")
            return

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
            
        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        print(full_df.head())  # Inspect the parsed DataFrame

        # Set the chunksize to the number of rows in the file / cpu cores available
        self.chunksize = len(full_df.index) / num_processes

        # Process chunks with guaranteed anomaly injection
        for chunk in [full_df[i:i+self.chunksize] for i in range(0, len(full_df), self.chunksize)]:
            if anomaly_settings:
                # If timestamps need adjustment, add start_time explicitly
                chunk[chunk.columns[0]] = chunk[chunk.columns[0]].apply(
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

        print("Inserting done!")

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
            
            case _:
                # Fileformat not supported
                return None
            # Add more fileformats here