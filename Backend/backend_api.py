import sys
from os import path
import socket
import json

HOST = "localhost"
PORT = 9524

DOC = """"python backend_api.py run-batch <model> <injection-method> <path-to-batch-file>"
starts anomaly detection of batch data from the given file with the given model and injection method

"python backend_api.py run-stream <model> <injection-method> <path-to-stream-file>" 
starts anomaly detection of stream data from the given file with the given model and injection method

"python backend_api.py change-model <model> <dataset-running>"
changes the model used for anomaly detection for the currently run batch or stream named <dataset-running> to <model>

"python backend_api.py change-injection <injection-method> <dataset-running>"
changes the injection method used for anomaly detection for the currently run batch or stream named <dataset-running> to <injection-method>

"python backend_api.py cancel <dataset-running>" 
cancels the currently running batch or stream named <dataset-running>

"python backend_api.py get-data <dataset-running>"
get all processed data from <dataset-running>, meaning just the data that has gone through our detection model
    
"python backend_api.py inject-anomaly <timestamps> <dataset-running>"    
injects anomalies in the data set running if manual injection is enabled, <timestamps> is a comma separated list of timestamps in seconds from now to inject anomalies at. (python backend_api.py inject-anomaly 10,20,30 system1 injects an anomaly at 10, 20 and 30 seconds from now)

"python backend_api.py get-running"
get all running datasets

"python backend_api.py get-models"
gets all available models for anomaly detection

"python backend_api.py get-injection-methods"
gets all available injection methods for anomaly detection
        
"python backend_api.py help"
prints this help message
"""

# Main function handles argument parsing when the API is invoked from the command line
def main(argv: list[str]) -> None:
    arg_len = len(argv)
    api = BackendAPI(HOST, PORT)
    # Start a batch job in the backend if the command is "run-batch"
    if argv[1] == "run-batch":
        if arg_len != 5:
            handle_error(1, "Invalid number of arguments")
        print(len([argv[2], argv[3], argv[4]]))
        result = api.run_batch(argv[2], argv[3], argv[4])

    # Start a stream job in the backend if the command is "run-stream"
    elif argv[1] == "run-stream":
        if arg_len != 5:
            handle_error(1, "Invalid number of arguments")
        result = api.run_stream(argv[2], argv[3], argv[4])
        
    # Change the model used for a running job if the command is "change-model"
    elif argv[1] == "change-model":
        if (arg_len != 4):
            handle_error(1, "Invalid number of arguments")
        result = api.change_model(argv[2], argv[3])

    # Change the injection method used for a running job if the command is "change-injection"
    elif argv[1] == "change-injection":
        if (arg_len != 4):
            handle_error(1, "Invalid number of arguments")
        result = api.change_method(argv[2], argv[3])
    
    # Get data from a running job if the command is "get-data", the backend will return data that has gone through the detection model
    elif argv[1] == "get-data":
        if (arg_len != 3):
            handle_error(1, "Invalid number of arguments")
        result = api.get_data(argv[2])
    
    # Inject anomalies into a running job if the command is "inject-anomaly"
    elif argv[1] == "inject-anomaly":
        if (arg_len != 4):
            handle_error(1, "Invalid number of arguments")
        timestamps = argv[2].split(',')
        result = api.inject_anomaly(timestamps, argv[3])

    # Print all running datasets if the command is "get-running"
    elif argv[1] == "get-running":
        if (arg_len != 2):
            handle_error(1, "Invalid number of arguments")
        result = api.get_running()

    # Cancel a running job if the command is "cancel"
    elif argv[1] == "cancel":
        if (arg_len != 3):
            handle_error(1, "Invalid number of arguments")
        result = api.cancel_job(argv[2])

    # Get all avaliable models for anomaly detection if the command is "get-models"
    elif argv[1] == "get-models":
        if (arg_len != 2):
            handle_error(1, "Invalid number of arguments")
        result = api.get_models()

    # Get all avaliable injection methods for anomaly detection if the command is "get-injection-methods"
    elif argv[1] == "get-injection-methods":
        if (arg_len != 2):
            handle_error(1, "Invalid number of arguments")
        result = api.get_injection_methods()

    # Print information about the backend API command line tool if the command is "help"
    elif argv[1] == "help":
        print(DOC)

    # Print an error message if the command is not recognized
    else: 
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
    def run_batch(self, model: str, injection_method: str, file_path: str) -> str:
        if not path.isfile(file_path):
            handle_error(2, "File not found")
        data = {
            "METHOD": "run-batch",
            "model": model,
            "injection_method": injection_method,
            "file_path": file_path
        }
        return self.__send_data(json.dumps(data))

    # Sends a request to the backend to start a stream job
    def run_stream(self, model: str, injection_method: str, file_path: str) -> str:
        if not path.isfile(path):
            handle_error(2, "File not found")
        data = {
            "METHOD": "run-stream",
            "model": model,
            "injection_method": injection_method,
            "file_path": file_path
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

    def __send_data(self, data: str) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sock.sendall(bytes(data, encoding="utf-8"))
        return json.loads(sock.recv(1024))

if __name__ == "__main__":
    main(sys.argv)