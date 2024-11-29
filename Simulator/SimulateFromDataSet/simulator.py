import time
import multiprocessing as mp          # To allow for fast insertion of the data
import pandas as pd                   
from pathlib import Path              
from API.db_interface import DBInterface as db    # The database API

file_path = './Datasets/system-1.csv'
x_speedup = 1
chunksize = 1

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

def process_entry(conn_params, table_name, chunk):
    """
    Processes a chunk of data by creating a DBInterface instance
    and inserting the chunk into the database.

    Args:
        chunk (pd.DataFrame): A chunk of data to be inserted.
    """
    db_instance = init_db(conn_params)
    db_instance.insert_data(table_name, chunk)

def filetype_csv(conn_params, file_path):
    # Get column names from the first row of the CSV file
    with open(file_path, 'r') as f:
        columns = f.readline().strip().split(',')
    
    create_table(conn_params, Path(file_path).stem, columns)

    dataindex = 1
    time_between_input = get_time_diffs_pandas(file_path).mean()
    print(f"Time between inputs: {time_between_input} seconds")
    print("Starting to insert!")
    for data in pd.read_csv(file_path, chunksize=chunksize):
        print(f"Inserting data {dataindex}")
        process_entry(conn_params, Path(file_path).stem, data)
        dataindex += 1
        time.sleep(time_between_input / x_speedup)

    print("Inserting done!")

def get_time_diffs_pandas(filename):
    """Reads a CSV file with pandas and calculates time differences between entries in the first column.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        pandas.Series: A pandas Series containing the time differences.
    """
    df = pd.read_csv(filename)
    time_diffs = df.iloc[:, 0].diff()  # Calculate differences between consecutive values in the first column
    return time_diffs

if __name__ == '__main__':
    conn_params = {
        "dbname": "mytimescaleDB",
        "user": "Anomdet",
        "passwd": "G5anomdet",
        "port": "5432",
        "host": "localhost"
    }
    file_extension = Path(file_path).suffix
    
    match file_extension:
        case ".csv":
            filetype_csv(conn_params, file_path)



"""
    TODO:
    Implement async simulaton to allow user to stop the simulator.
    Implement reading the filepath from the frontend.
    Implement anomaly insertion.
"""