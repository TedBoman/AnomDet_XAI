import sys
import os
import socket
import json
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('BACKEND_HOST', '127.0.0.1')
PORT = int(os.getenv('BACKEND_PORT', 9524))

DOC = """"python api.py run-batch <model> <injection-method> <dataset>"
starts anomaly detection of batch data from the given file with the given model and injection method

"python api.py run-stream <model> <injection-method> <dataset>" 
starts anomaly detection of stream data from the given file with the given model and injection method

"python api.py change-model <model> <dataset-running>"
changes the model used for anomaly detection for the currently run batch or stream named <dataset-running> to <model>

"python api.py change-injection <injection-method> <dataset-running>"
changes the injection method used for anomaly detection for the currently run batch or stream named <dataset-running> to <injection-method>

"python api.py cancel <dataset-running>" 
cancels the currently running batch or stream named <dataset-running>

"python api.py get-data <timestamp> <dataset-running>"
get all processed data from <dataset-running>, meaning just the data that has gone through our detection model
    
"python api.py inject-anomaly <timestamps> <dataset-running>"    
injects anomalies in the data set running if manual injection is enabled, <timestamps> is a comma separated list of timestamps in seconds from now to inject anomalies at. (python api.py inject-anomaly 10,20,30 system1 injects an anomaly at 10, 20 and 30 seconds from now)

"python api.py get-running"
get all running datasets

"python api.py get-models"
gets all available models for anomaly detection

"python api.py get-injection-methods"
gets all available injection methods for anomaly detection

"python api.py get-datasets"
gets all available datasets

"python api.py upload-dataset <dataset-file-path>"
uploads a dataset to the backend by adding the file to the Dataset directory
        
"python api.py help"
prints this help message
"""

# Main function handles argument parsing when the API is invoked from the command line
def main(argv: list[str]) -> None:
    arg_len = len(argv)
    api = BackendAPI(HOST, PORT)
    match argv[1]:
        # Start a batch job in the backend if the command is "run-batch"
        case "run-batch":
            if arg_len != 5:
                handle_error(1, "Invalid number of arguments")
            result = api.run_batch(argv[2], argv[3], argv[4])

        # Start a stream job in the backend if the command is "run-stream"
        case "run-stream":
            if arg_len != 5:
                handle_error(1, "Invalid number of arguments")
            result = api.run_stream(argv[2], argv[3], argv[4])
            
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
            if (arg_len != 3):
                handle_error(1, "Invalid number of arguments")
            result = api.get_data(argv[2])
        
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

        # Upload a dataset to the backend if the command is "upload-dataset"
        case "upload-dataset":
            if (arg_len != 3):
                handle_error(1, "Invalid number of arguments")
            result = api.upload_dataset(argv[2])

        # Print information about the backend API command line tool if the command is "help"
        case "help":
            print(DOC)

        # Print an error message if the command is not recognized
        case _: 
            handle_error(3, f'argument "{argv[1]}" not recognized as a valid command')

    # Print return messgage in terminal when API is used by the command line tool
    if argv[1] != "help":
        print(f'Recieved from backend: {result}')
        
def handle_error(code: int, message: str) -> None:
        print(message)
        exit(code)        
class BackendAPI:
    # Constructor setting host adress and port for the the backend container
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    # Sends a request to the backend to start a batch job
    def run_batch(self, model: str, injection_method: str, dataset: str) -> str:
        data = {
            "METHOD": "run-batch",
            "model": model,
            "injection_method": injection_method,
            "dataset": dataset
        }
        return self.__send_data(json.dumps(data))

    # Sends a request to the backend to start a stream job
    def run_stream(self, model: str, injection_method: str, dataset: str) -> str:
        data = {
            "METHOD": "run-stream",
            "model": model,
            "injection_method": injection_method,
            "dataset": dataset
        }
        return self.__send_data(json.dumps(data))

    # Sends a request to the backend to change the model used for a running job
    def change_model(self, model: str, name: str) -> str:
        data = {
            "METHOD": "change-model",
            "model": model,
            "job_name": name
        }
        return self.__send_data(json.dumps(data))

    # Sends a request to the backend to change the injection method used for a running job
    def change_method(self, injection_method: str, name: str) -> str:
        data = {
            "METHOD": "change-method",
            "injection-method": injection_method,
            "job_name": name
        }
        return self.__send_data(json.dumps(data))


    def get_data(self, name: str) -> str:
        data = {
            "METHOD": "get-data",
            "job_name": name
        }
        return self.__send_data(json.dumps(data))

    def inject_anomaly(self, timestamps: list[int], name: str) -> str:
        data = {
            "METHOD": "inject-anomaly",
            "timestamps": timestamps,
            "job_name": name
        }
        return self.__send_data(json.dumps(data))

    def get_running(self) -> str:
        data = {
            "METHOD": "get-running"
        }
        return self.__send_data(json.dumps(data))
    
    def cancel_job(self, name: str) -> str:
        data = {
            "METHOD": "cancel-job",
            "job_name": name
        }
        return self.__send_data(json.dumps(data))
    
    def get_models(self) -> str:
        data = {
            "METHOD": "get-models"
        }
        return self.__send_data(json.dumps(data))
    
    def get_injection_methods(self) -> str:
        data = {
            "METHOD": "get-injection-methods"
        }
        return self.__send_data(json.dumps(data))
    
    def get_datasets(self) -> str:
        data = {
            "METHOD": "get-datasets"
        }
        return self.__send_data(json.dumps(data))
    
    def upload_dataset(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            handle_error(2, "File not found")
        data = {
            "METHOD": "upload-dataset",
            "file_path": file_path
        }
        return self.__send_data(json.dumps(data))

    def __send_data(self, data: str) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sock.sendall(bytes(data, encoding="utf-8"))
        return json.loads(sock.recv(1024))

if __name__ == "__main__":
    main(sys.argv)