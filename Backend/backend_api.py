import sys
import os
import socket
import json
from time import sleep
from dotenv import load_dotenv

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('BACKEND_PORT'))

DOC = """"python backend_api.py run-batch <model> <injection-y/n> <dataset> <name>"
starts anomaly detection of batch data from the given file with the given model and asks for injection details if injectin is "y"

"python backend_api.py run-stream <model> <injection-y/n> <dataset> <name> <speedup-optional>" 
starts anomaly detection of stream data from the given file with the given model and asks for injection details if injectin is "y". Speedup is "1" for 1x speed, "2" for 2x speed etc.

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
    arg_len = len(argv)
    api = BackendAPI(HOST, PORT)
    match argv[1]:
        # Start a batch job in the backend if the command is "run-batch"
        case "run-batch":
            if arg_len != o:
                handle_error(1, "Invalid number of arguments")
            if argv[3] == "y":
                anomaly_type = input("Enter the type of anomaly: ")
                timestamp = input("Enter the timestamp to start anomaly: ")
                magnitude = input("Enter the magnitude of the anomaly: ")
                percentage = input("Enter the percentage of data: ")
                duration = input("Enter the duration of the anomalies: ")
                columns_string = input("Enter the columns to inject anomalies into, as a comma separated list (a,b,c,d,...): ")
                inj_params = {
                    "anomaly_type": anomaly_type,
                    "timestamp": timestamp,
                    "magnitude": magnitude,
                    "percentage": percentage,
                    "duration": duration,
                    "columns": columns_string.split(',')
                }
                api.run_batch(argv[2], argv[4], argv[5], inj_params)
            else:
                api.run_batch(argv[2], argv[4], argv[5])

        # Start a stream job in the backend if the command is "run-stream"
        case "run-stream":
            if arg_len != 6 and arg_len != 7:
                handle_error(1, "Invalid number of arguments")
            if argv[3] == "y":
                anomaly_type = input("Enter the type of anomaly: ")
                timestamp = input("Enter the timestamp to start anomaly: ")
                magnitude = input("Enter the magnitude of the anomaly: ")
                percentage = input("Enter the percentage of data: ")
                duration = input("Enter the duration of the anomalies: ")
                columns_string = input("Enter the columns to inject anomalies into, as a comma separated list (a,b,c,d,...): ")
                inj_params = {
                    "anomaly_type": anomaly_type,
                    "timestamp": timestamp,
                    "magnitude": magnitude,
                    "percentage": percentage,
                    "duration": duration,
                    "columns": columns_string.split(',')
                }
                if arg_len == 7:
                    api.run_stream(argv[2], argv[4], argv[5], speedup=argv[6], inj_params=inj_params)
                else:
                    api.run_stream(argv[2], argv[4], argv[5], inj_params=inj_params)
            else:
                if arg_len == 7:
                    api.run_stream(argv[2], argv[4], argv[5], speedup=argv[6])
                else:
                    api.run_stream(argv[2], argv[4], argv[5])
            
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
    def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
        data = {
            "METHOD": "run-batch",
            "model": model,
            "dataset": dataset,
            "name": name
        }
        if inj_params:
            data["injection_params"] = inj_params

        self.__send_data(data, response=False)

    # Sends a request to the backend to start a stream job
    def run_stream(self, model: str,dataset: str, name: str, speedup: str=None, inj_params: dict=None) -> None:
        data = {
            "METHOD": "run-stream",
            "model": model,
            "dataset": dataset,
            "name": name
        }
        if inj_params:
            data["injection_params"] = inj_params
        if speedup:
            data["speedup"] = speedup
            
        self.__send_data(data, response=False)

    # Sends a request to the backend to change the model used for a running job
    def change_model(self, model: str, name: str) -> None:
        data = {
            "METHOD": "change-model",
            "model": model,
            "job_name": name
        }
        self.__send_data(data, response=False)

    # Sends a request to the backend to change the injection method used for a running job
    def change_method(self, injection_method: str, name: str) -> None:
        data = {
            "METHOD": "change-method",
            "injection-method": injection_method,
            "job_name": name
        }
        self.__send_data(data, response=False)

    # Requests each row of data of a running job from timestamp and forward
    def get_data(self, timestamp: str, name: str) -> str:
        data = {
            "METHOD": "get-data",
            "timestamp": timestamp,
            "job_name": name
        }
        return self.__send_data(data)

    # Injects anomalies into a running job
    def inject_anomaly(self, timestamps: list[int], name: str) -> None:
        data = {
            "METHOD": "inject-anomaly",
            "timestamps": timestamps,
            "job_name": name
        }
        self.__send_data(data, response=False)

    # Get all running jobs
    def get_running(self) -> str:
        data = {
            "METHOD": "get-running"
        }
        return self.__send_data(data)
    
    # Cancels a running job, deletes the data and stops the anomaly detection
    def cancel_job(self, name: str) -> None:
        data = {
            "METHOD": "cancel-job",
            "job_name": name
        }
        self.__send_data(data, response=False)
    
    # Get all available models for anomaly detection
    def get_models(self) -> str:
        data = {
            "METHOD": "get-models"
        }
        return self.__send_data(data)
    
    # Get all available anomaly injection methods
    def get_injection_methods(self) -> str:
        data = {
            "METHOD": "get-injection-methods"
        }
        return self.__send_data(data)
    
    # Get all available datasets
    def get_datasets(self) -> str:
        data = {
            "METHOD": "get-datasets"
        }
        return self.__send_data(data)
    
    # Uploads a complete dataset to the backend
    def import_dataset(self, file_path: str, timestamp_column: str) -> None:
        if not os.path.isfile(file_path):
            handle_error(2, "File not found")

        file = open(file_path, "r")
        file_content = file.read()
        data = {
            "METHOD": "import-dataset",
            "name": os.path.basename(file_path),
            "timestamp_column": timestamp_column,
            "file_content": file_content
        }
        
        self.__send_data(data, response=False)

    # Initates connection to backend and sends json data through the socket
    def __send_data(self, data: str, response: bool=True) -> str:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))

            # Send two messages through the same connection if the method is import-dataset
            if data["METHOD"] == "import-dataset":
                file_content = data["file_content"]
                del data["file_content"]
                data = json.dumps(data)
                sock.sendall(bytes(data, encoding="utf-8"))
                sleep(0.5)
                sock.sendall(bytes(file_content, encoding="utf-8"))
            else:
                data = json.dumps(data)
                sock.sendall(bytes(data, encoding="utf-8"))
            if response:
                data = sock.recv(1024)
                return data
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main(sys.argv)