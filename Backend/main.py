# main.py

# Third-Party
import threading
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from typing import Union, List, Optional, Dict

# Custom
from Simulator.SimulateFromDataSet.simulator import Simulator
from Simulator.BatchImport.batchimport import BatchImporter
from Backend.talk_to_backend import TalkToBackend as ttb
from Backend import talk_to_backend as ttb2

class test_sim:
    def process_file(self, file_path, conn_params, simulation_type, anomaly_settings, start_time ,speedup: int = 1):
        """Processes a single file based on the specified use case."""
        print("Starting to process file!")
        match simulation_type:
            case "stream":
                print("Starting stream job...")
                file_extension = Path(file_path).suffix
                sim = Simulator(file_path, file_extension, start_time, x_speedup=speedup)
                match file_extension:
                    case ".csv":
                        sim.start_simulation(conn_params, anomaly_settings)
            case "batch":
                print("Starting batch job...")
                file_extension = Path(file_path).suffix
                sim = BatchImporter(file_path, file_extension, start_time)
                match file_extension:
                    case ".csv":
                        sim.start_simulation(conn_params, anomaly_settings)

    def start(self):
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

        if testMessage:

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

                thread = threading.Thread(target=self.process_file, 
                                        args=(testMessage.filepath, conn_params, testMessage.simulation_type, testMessage.anomaly_settings, start_time, testMessage.speedup))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()


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