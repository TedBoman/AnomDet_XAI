# batchimport.py

import multiprocessing as mp
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
from DBAPI import utils as ut
import datetime

class BatchImporter:

    def __init__(self, file_path, start_time, chunksize=100):
        self.file_path = file_path
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

    def process_chunk(self, conn_params, table_name, chunk, isAnomaly=False):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
            table_name: The name of the table.
            chunk (pd.DataFrame): A chunk of data to be inserted.
            isAnomaly (bool): Indicates if the chunk contains an anomaly.
        """
        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, chunk, isAnomaly)  # Use isAnomaly flag

    def inject_anomalies_into_chunk(self, chunk, anomaly_settings):
        """
        Inject anomalies into a chunk of data based on the given settings.
        """
        try:
            print("Injecting anomalies into chunk...")
            print(f"Chunk shape: {chunk.shape}")
            print(f"Anomaly settings: {anomaly_settings}")

            if not isinstance(chunk, pd.DataFrame):
                raise ValueError("Expected chunk to be a pandas DataFrame.")

            injector = TimeSeriesAnomalyInjector() 
            chunk_start_time = chunk['timestamp'].min()  # Adjust to actual timestamp column
            chunk_end_time = chunk['timestamp'].max()

            anomaly_applied = False  # Flag to track if any anomaly was applied

            for anomaly in anomaly_settings:

                start_time = anomaly['timestamp']  # Start time is already an absolute timestamp
                end_time = start_time + ut.parse_duration(anomaly.get('duration', '0s'))  # Calculate end time

                # Check if the chunk overlaps with the anomaly’s time range
                if (chunk_start_time <= end_time) and (chunk_end_time >= start_time):
                    chunk = injector.inject_anomaly(chunk, anomaly)
                    anomaly_applied = True
                    print(f"Anomaly '{anomaly['anomaly_type']}' applied from {start_time} to {end_time}")
                else:
                    print(f"No overlap for anomaly '{anomaly['anomaly_type']}'.")

            # Debug: Log the result of anomaly injection
            print(f"Anomaly applied: {anomaly_applied}")
            return chunk, anomaly_applied

        except Exception as e:
            print(f"Error injecting anomalies into chunk: {e}")
            return chunk, False

    def filetype_csv(self, conn_params, anomaly_settings=None):
        """
        Processes a CSV file, injects anomalies, and inserts the data into the database.
        Ensures consistent anomaly injection across chunks.
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
                if not isinstance(setting['timestamp'], pd.Timestamp):
                    setting['timestamp'] = self.start_time + pd.Timedelta(seconds=setting['timestamp'])

        # Create a list to store results from async processes
        results = []

        full_df = pd.read_csv(self.file_path)

        # Convert the first column (assume it's timestamps in seconds)
        full_df[full_df.columns[0]] = self.start_time + pd.to_timedelta(
            full_df[full_df.columns[0]].astype(float), unit='s'
        )

        # Drop rows with invalid timestamps
        full_df = full_df.dropna(subset=[full_df.columns[0]])

        print(full_df.head())  # Inspect the parsed DataFrame

        # Process chunks with guaranteed anomaly injection
        for chunk in [full_df[i:i+self.chunksize] for i in range(0, len(full_df), self.chunksize)]:
            anomaly_applied = False

            if anomaly_settings:
                # If timestamps need adjustment, add start_time explicitly
                chunk[chunk.columns[0]] = chunk[chunk.columns[0]].apply(
                    lambda x: self.start_time + pd.Timedelta(seconds=x.timestamp())
                    if isinstance(x, datetime.datetime) and x < self.start_time
                    else x
                )

                # Inject anomalies across chunk boundaries
                chunk, anomaly_applied = self.inject_anomalies_into_chunk(chunk, anomaly_settings)

            # Use apply_async and collect results
            result = pool.apply_async(
                self.process_chunk,
                args=(conn_params, table_name, chunk, anomaly_applied),
            )
            results.append(result)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        # Optionally, check for any exceptions in the results
        for result in results:
            result.get()  # This will raise any exceptions that occurred in the process

        print("Inserting done!")
