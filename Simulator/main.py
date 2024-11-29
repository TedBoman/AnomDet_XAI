# main.py

from pathlib import Path
from SimulateFromDataSet.simulator import Simulator
from BatchImport.batchimport import BatchImporter

use = "batch"

if __name__ == '__main__':
    file_path = './Datasets/system-1.csv'  # Or get this from the frontend
    conn_params = {
        "dbname": "mytimescaleDB",
        "user": "Anomdet",
        "passwd": "G5anomdet",
        "port": "5432",
        "host": "localhost"
    }

    match use:
        case "sim":
            sim = Simulator(file_path)  # Create a Simulator instance

            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    sim.filetype_csv(conn_params)
        case "batch":
            importer = BatchImporter(file_path)

            file_extension = Path(file_path).suffix
            match file_extension:
                case ".csv":
                    importer.filetype_csv(conn_params)