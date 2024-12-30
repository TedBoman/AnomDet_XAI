# main.py

# Third-Party
from pathlib import Path
import pandas as pd
import os

# Custom
from Simulator.SimulateFromDataSet.simulator import Simulator
from Simulator.BatchImport.batchimport import BatchImporter
from Simulator.DBAPI import type_classes as tc
from Simulator.DBAPI.debug_utils import DebugLogger as dl

DEFAULT_PATH = './Datasets/'

class SimulatorEngine:
    def process_file(self, file_path, conn_params, simulation_type, anomaly_settings, start_time, speedup: int = 1, table_name = None):
        """Processes a single file based on the specified use case."""
        dl.debug_print("Starting to process file!")
        match simulation_type:
            case "stream":
                dl.debug_print("Starting stream job...")
                try:
                    file_extension = Path(file_path).suffix
                    sim = Simulator(file_path, file_extension, start_time, x_speedup=speedup)
                    match file_extension:
                        case ".csv":
                            sim.start_simulation(conn_params, anomaly_settings, table_name)
                except Exception as e:
                    dl.debug_print(f"Error: {e}")
            case "batch":
                dl.debug_print("Starting batch job...")
                file_extension = Path(file_path).suffix
                sim = BatchImporter(file_path, file_extension, start_time, 5)
                match file_extension:
                    case ".csv":
                        sim.start_simulation(conn_params, anomaly_settings, table_name)

    def main(self, db_conn_params, job, debug=True):
        # Set debug mode once for all files
        dl.set_debug(debug)  # or False to disable debug prints

        # Check if the path exists
        if os.path.isfile(job.filepath):
            # Filepath is valid, do nothing
            pass
        else:
            # Prepend the default path to the filename
            job.filepath = os.path.join(DEFAULT_PATH, os.path.basename(job.filepath))

        if job:
            # Convert anomaly settings timestamps to datetime objects
            if isinstance(job, (tc.Job)) and job.anomaly_settings:
                for setting in job.anomaly_settings:
                    # Access the timestamp attribute of the AnomalySetting object
                    setting.timestamp = pd.to_datetime(setting.timestamp, unit='s')

                self.process_file(job.filepath, db_conn_params, job.simulation_type, job.anomaly_settings, pd.to_timedelta(0), job.speedup, job.table_name if job.table_name else None)
            else:

                self.process_file(job.filepath, db_conn_params, job.simulation_type, job.anomaly_settings, pd.to_timedelta(0), job.speedup, job.table_name if job.table_name else None)

