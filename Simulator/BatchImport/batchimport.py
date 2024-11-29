# batchimport.py

import multiprocessing as mp
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db

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

    def process_chunk(self, conn_params, table_name, chunk):
        """
        Processes a chunk of data by creating a DBInterface instance
        and inserting the chunk into the database.

        Args:
            conn_params: The parameters needed to connect to the database.
            table_name: The name of the table.
            chunk (pd.DataFrame): A chunk of data to be inserted.
        """
        db_instance = self.init_db(conn_params)
        db_instance.insert_data(table_name, chunk)

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
        self.create_table(conn_params, Path(self.file_path).stem, columns)

        print("Starting to insert!")
        # For every chunksize rows in the file, we start a process that inserts those rows
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize):
            pool.apply_async(self.process_chunk, args=(conn_params, Path(self.file_path).stem, chunk))
        print("Inserting done!")

        # Wait for all the processes to finish
        pool.close()
        pool.join()