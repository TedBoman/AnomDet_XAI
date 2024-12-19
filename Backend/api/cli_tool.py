import sys
import os
from api import BackendAPI

from run_batch import run_batch
from run_stream import run_stream
from dotenv import load_dotenv

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('BACKEND_PORT'))

DOC = """"python backend_api.py run-batch"
starts anomaly detection of batch data after user has been prompted to enter details of the job

"python backend_api.py run-stream" 
starts anomaly detection of stream data after user has been prompted to enter details of the job

"python backend_api.py change-model <model> <name>"
changes the model used for anomaly detection for the currently run batch named <name> to <model>

"python backend_api.py change-injection <injection-method> <name>"
changes the injection method used for anomaly detection for the currently run batch named <name> to <injection-method>

"python backend_api.py get-data <timestamp> <name>"
get all processed data from <name>, meaning just the data that has gone through our detection model. <timestamp> allows for filtering of data
    
"python backend_api.py inject-anomaly <timestamps> <name>"    
injects anomalies in the data set <name> if manual injection is enabled, <timestamps> is a comma separated list of timestamps in seconds from now to inject anomalies at. (python backend_api.py inject-anomaly 10,20,30 system1 injects an anomaly at 10, 20 and 30 seconds from now)

"python backend_api.py get-running"
get all running datasets

"python backend_api.py cancel <name>" 
cancels the currently running batch or stream named <name>

"python backend_api.py get-models"
gets all available models for anomaly detection

"python backend_api.py get-injection-methods"
gets all available injection methods for anomaly detection

"python backend_api.py get-datasets"
gets all available datasets

"python backend_api.py import-dataset <dataset-file-path> <timestamp-column-name>"
uploads a dataset to the backend by adding the file to the Dataset directory
        
"python backend_api.py help"
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
            
        # Change the model used for a running job if the command is "change-model"
        case "change-model":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            result = api.change_model(argv[2], argv[3])

        # Change the injection method used for a running job if the command is "change-injection"
        case "change-injection":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            result = api.change_method(argv[2], argv[3])
        
        # Get data from a running job if the command is "get-data", the backend will return data that has gone through the detection model
        case "get-data":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            result = api.get_data(argv[2], argv[3])
        
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
        case "cancel":
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

        # Upload a dataset to the backend if the command is "import-dataset"
        case "import-dataset":
            if (arg_len != 4):
                handle_error(1, "Invalid number of arguments")
            result = api.import_dataset(argv[2], argv[3])

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