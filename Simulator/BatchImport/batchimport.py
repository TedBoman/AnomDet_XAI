import os
import multiprocessing as mp          # To allow for fast insertion of the data
import pandas as pd                   
from pathlib import Path              
from API.db_interface import DBInterface as db    # The database API

file_name = './Datasets/system-1.csv'

def init_db(conn_params) -> db:
    """
    Returns an instance of the database interface API.

    Args:
        conn_params:    A dictionary containing the parameters needed
                        to connect to the timeseries database.
    """
    db_instance = db(conn_params)
    return db_instance

def create_table(conn_params, tb_name, columns):
    """
    Creates a table in the timeseries database.

    Args: 
        conn_params: The parameters needed to connect to the database.
        tb_name:     The name of the table.
        columns:     The columns for the table.
    """
    db_instance = init_db(conn_params)
    db_instance.create_table(tb_name, columns)

def process_chunk(conn_params, table_name, columns, chunk):
    """
    Processes a chunk of data by creating a DBInterface instance
    and inserting the chunk into the database.

    Args:
        chunk (pd.DataFrame): A chunk of data to be inserted.
    """
    db_instance = init_db(conn_params)
    db_instance.insert_data(table_name, chunk)

if __name__ == '__main__':
    conn_params = {
        "dbname": "mytimescaleDB",
        "user": "Anomdet",
        "passwd": "G5anomdet",
        "port": "5432",
        "host": "localhost"
    }

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    # Get column names from the first row of the CSV file
    with open(file_name, 'r') as f:
        columns = f.readline().strip().split(',')
    
    create_table(conn_params, Path(file_name).stem, columns)

    chunksize = 20
    print("Starting to insert!")
    for chunk in pd.read_csv(file_name, chunksize=chunksize):
        pool.apply_async(process_chunk, args=(conn_params, Path(file_name).stem, columns, chunk))
    print("Inserting done!")
    pool.close()
    pool.join()
