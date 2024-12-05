# main.py

import threading
from pathlib import Path
from SimulateFromDataSet.simulator import Simulator
from BatchImport.batchimport import BatchImporter
import multiprocessing as mp


use = "batch"

def process_file(file_path, conn_params, use, anomaly_settings, speedup: int = 1):
    """Processes a single file based on the specified use case."""
    match use:
        case "sim":
            sim = Simulator(file_path, x_speedup=speedup)
            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    sim.filetype_csv(conn_params)
        case "batch":
            importer = BatchImporter(file_path)
            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    importer.filetype_csv(conn_params, anomaly_settings)

def listen_to_front():
    return

def main(argv: list[str]):
    conn_params = {
        "dbname": "TSdatabase",
        "user": "Anomdet",
        "passwd": "G5anomdet",
        "port": "5432",
        "host": "host.docker.internal"
    }

    file_paths = [  # List of file paths to process
        './Datasets/test_system.csv',
    ]

    anomaly_settings = {
        "anomaly_type": "lowered",
        "timestamp": 210,
        "magnitude": 2,
        "percentage": 5,
        "columns": ["load-5m", "load-1m"],
    }

    threads = []
    for file_path in file_paths:
        thread = threading.Thread(target=process_file, args=(file_path, conn_params, use, anomaly_settings))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main([])