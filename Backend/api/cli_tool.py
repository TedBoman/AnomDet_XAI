import sys
import os
from datetime import datetime, timezone
from api import BackendAPI

from run_batch import run_batch
from run_stream import run_stream
from dotenv import load_dotenv

dotenv_path = "../../Docker/.env"

load_dotenv(dotenv_path)
HOST = 'localhost'
PORT = int(os.getenv('BACKEND_PORT'))

DOC = """"python api.py run-batch"
starts anomaly detection of batch data after user has been prompted to enter details of the job

"python api.py run-stream" 
starts anomaly detection of stream data after user has been prompted to enter details of the job

"python api.py get-data <timestamp> <name>"
get all processed data from <name>, meaning just the data that has gone through our detection model. <timestamp> allows for filtering of data. <timestamp> is in seconds from epoch.

"python api.py get-running"
get all running datasets

"python api.py cancel-job <name>" 
cancels the currently running batch or stream named <name>

"python api.py get-models"
gets all available models for anomaly detection

"python api.py get-injection-methods"
gets all available injection methods for anomaly detection

"python api.py get-datasets"
gets all available datasets

"python api.py get-all-jobs"
gets all started and/or running jobs

"python api.py get-columns <name>"

"python api.py import-dataset <dataset-file-path> <timestamp-column-name>"
uploads a dataset to the backend by adding the file to the Dataset directory
        
"python api.py help"
prints this help message
"""

# Main function handles argument parsing when the API is invoked from the command line
def main(argv: list[str]) -> None:
    result = None
    arg_len = len(argv)
    api = BackendAPI(HOST, PORT)
    match argv[1]:
        # Start a batch job in the backend if the command is "run-batch"
        case "run-batch":
            if arg_len != 2:
                handle_error(1, "Invalid number of arguments")

            # Makes user input and sends request to the backend
            run_batch(api)
            
        # Start a stream job in the backend if the command is "run-stream"
        case "run-stream":
            if arg_len != 2:
                handle_error(1, "Invalid number of arguments")
            
            # Makes user input and sends request to the backend
            run_stream(api)
        
        # Get data from a running job if the command is "get-data", the backend will return data that has gone through the detection model
        case "get-data":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            result = api.get_data(datetime.fromtimestamp(argv[2]), argv[3])
        
        # Inject anomalies into a running job if the command is "inject-anomaly"
        case "inject-anomaly":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            timestamps = argv[2].split(',')
            result = api.inject_anomaly(timestamps, argv[3])

        # Print all running datasets if the command is "get-running"
        case "get-running":
            if (arg_len != 2):
                handle_error(1, "Invalid number of arguments")
            result = api.get_running()

        # Cancel a running job if the command is "cancel"
        case "cancel-job":
            if (arg_len != 3):
                handle_error(1, "Invalid number of arguments")
            result = api.cancel_job(argv[2])

        # Get all avaliable models for anomaly detection if the command is "get-models"
        case "get-models":
            if (arg_len != 2):
                handle_error(1, "Invalid number of arguments")
            result = api.get_models()

        # Get all avaliable injection methods for anomaly detection if the command is "get-injection-methods"
        case "get-injection-methods":
            if (arg_len != 2):
                handle_error(1, "Invalid number of arguments")
            result = api.get_injection_methods()

        # Get all avaliable datasets if the command is "get-datasets"
        case "get-datasets":
            if (arg_len != 2):
                handle_error(1, "Invalid number of arguments")
            result = api.get_datasets()
        
        # Get all started and/or running jobs
        case "get-all-jobs":
            if (arg_len != 2):
                handle_error(1, "Invalid number of arguments")
            result = api.get_all_jobs()

        # Get columns of a running job
        case "get-columns":
            if (arg_len != 3):
                handle_error(1, "Invalid number of arguments")
            result = api.get_columns(argv[2])

        # Upload a dataset to the backend if the command is "import-dataset"
        case "import-dataset":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            api.import_dataset(argv[2], argv[3])

        # Print information about the backend API command line tool if the command is "help"
        case "help":
            print(DOC)

        # Print an error message if the command is not recognized
        case _: 
            handle_error(3, f'argument "{argv[1]}" not recognized as a valid command')

    # Print return messgage in terminal when API is used by the command line tool
    if argv[1] != "help" and result:
        print(f'Recieved from backend: {result}')
        
def handle_error(code: int, message: str) -> None:
        print(message)
        exit(code) 

if __name__ == "__main__":
    main(sys.argv)