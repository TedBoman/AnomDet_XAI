# batchimport.py

import multiprocessing as mp
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector

class BatchImporter:

    def __init__(self, file_path, chunksize=20):
        self.file_path = file_path
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
    
    def check_for_message():
        return False

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
            if "already exists" in str(e):
                i = 1
                new_table_name = f"{tb_name}_{i}"
                while True:
                    try:
                        db_instance.create_table(new_table_name, columns)
                        return new_table_name  # Return the new table name
                    except Exception as e:
                        if "already exists" in str(e):
                            i += 1
                            new_table_name = f"{tb_name}_{i}"
                        else:
                            raise e
            else:
                raise e

    def process_chunk(self, conn_params, table_name, chunk, inject: bool = False):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
            table_name: The name of the table.
            chunk (pd.DataFrame): A chunk of data to be inserted.
        """

        if(inject):
            # Create an anomaly injector
            anomaly_injector = TimeSeriesAnomalyInjector()

            # Inject anomalies before inserting into database
            chunk_with_anomalies = anomaly_injector.inject_anomaly(
                chunk, 
                anomaly_type='spike',  # Choose anomaly type
                percentage=0.05,        # Percentage of points to modify
                magnitude=1.5           # Intensity of anomalies
            )

            db_instance = self.init_db(conn_params)
            db_instance.insert_data(table_name, chunk_with_anomalies, True)
        else:
            db_instance = self.init_db(conn_params)
            db_instance.insert_data(table_name, chunk, False)

    def filetype_csv(self, conn_params):
        """
        Takes a filepath to a CSV file, divides it into chunks, and inserts them into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
        """
        # Get the number of cores the PC has
        num_processes = mp.cpu_count()
        # Create one spot in the pool for each core
        pool = mp.Pool(processes=num_processes)

        # Get column names from the first row of the CSV file
        with open(self.file_path, 'r') as f:
            columns = f.readline().strip().split(',')
        
        # Creates a table in the database with the filename as table name and the correct columns
        table_name = self.create_table(conn_params, Path(self.file_path).stem, columns)

        print("Starting to insert!")
        # For every chunksize rows in the file, we start a process that inserts those rows
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize):
            pool.apply_async(self.process_chunk, args=(conn_params, table_name, chunk))
        print("Inserting done!")

        # Wait for all the processes to finish
        pool.close()
        pool.join()