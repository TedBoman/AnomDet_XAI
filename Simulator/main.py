# main.py

# Third-Party
import threading
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from typing import Union, List, Optional, Dict

# Custom
from SimulateFromDataSet.simulator import Simulator
from BatchImport.batchimport import BatchImporter
from DBAPI.talk_to_backend import TalkToBackend as ttb
from DBAPI import talk_to_backend as ttb2

use = "batch"

def process_file(file_path, conn_params, use, anomaly_settings, start_time ,speedup: int = 1):
    """Processes a single file based on the specified use case."""
    match use:
        case "sim":
            sim = Simulator(file_path, x_speedup=speedup)
            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    sim.filetype_csv(conn_params)
        case "batch":
            importer = BatchImporter(file_path, start_time)
            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    importer.filetype_csv(conn_params, anomaly_settings)

def listen_to_front():
    t = ttb()
    while True:
        message_received = t.ReadFromBackEnd()

        if message_received:
            pass

def main(argv: list[str]):
    conn_params = {
        "dbname": "TSdatabase",
        "user": "Anomdet",
        "passwd": "G5anomdet",
        "port": "5432",
        "host": "host.docker.internal"
    }

    t = ttb()
    testMessage = t.Test()

    print(f"{t}")

    print(f"Filepath: {testMessage.filepath}")
    threads = []
    # Read the CSV file to get the start time 
    df = pd.read_csv(testMessage.filepath)
    start_time = pd.to_datetime(df.iloc[0, 0])  # Get start time as datetime object

    # Convert anomaly settings timestamps to datetime objects
    if isinstance(testMessage, (ttb2.Message)) and testMessage.anomaly_settings:
        for setting in testMessage.anomaly_settings:
            # Access the timestamp attribute of the AnomalySetting object
            setting.timestamp = pd.to_timedelta(setting.timestamp, unit='s') + start_time  

        thread = threading.Thread(target=process_file, 
                                args=(testMessage.filepath, conn_params, use, testMessage.anomaly_settings, start_time))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main([])

    """
            file_paths = [  # List of file paths to process
        './Datasets/test_system.csv',
    ]

    anomaly_settings = [
        {
        "anomaly_type": "lowered",
        "timestamp": 630,
        "magnitude": 2,
        "percentage": 100,
        "duration": "5m",
        "columns": ["load-5m", "load-1m"],},
        #{
        #"anomaly_type": "spike",
        #"timestamp": 210,
        #"magnitude": 2,
        #"percentage": 50,
        #"duration": "2m",
        #"columns": ["load-5m", "load-1m"],}
        #Dic list
    ]
    """