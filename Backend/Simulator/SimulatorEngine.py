# main.py

# Third-Party
from pathlib import Path
import pandas as pd

# Custom
from Simulator.SimulateFromDataSet.simulator import Simulator
from Simulator.BatchImport.batchimport import BatchImporter
from Simulator.DBAPI import talk_to_backend as ttb2

class SimulatorEngine:
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
                sim = BatchImporter(file_path, file_extension, start_time, 5)
                match file_extension:
                    case ".csv":
                        sim.start_simulation(conn_params, anomaly_settings)

    def main(self, job):
        conn_params = {
            "dbname": "mytimescaleDB",
            "user": "Anomdet",
            "passwd": "G5anomdet",
            "port": "5432",
            "host": "host.docker.internal"
        }

        if job:
            # Convert anomaly settings timestamps to datetime objects
            if isinstance(job, (ttb2.Message)) and job.anomaly_settings:
                for setting in job.anomaly_settings:
                    # Access the timestamp attribute of the AnomalySetting object
                    setting.timestamp = pd.to_timedelta(setting.timestamp, unit='s') + 0  

                self.process_file(job.filepath, conn_params, job.simulation_type, job.anomaly_settings, 0, job.speedup)