# Backend API and CLI-tool documentation

## CLI-tool

We've made it possible to send requests to the backend with the backend api through the command-line. This is made possible by the invoking "./Backend/api/cli_tool.py". The main function parses the command-line arguments and calls, handles errors and calls appropiate function sending data to the backend. Run "python cli_tool.py help" for CLI-tool documentation.

## BackendAPI

Class designed to serve all requests/responses to and from the backend. It is initialized with a host and port sent to the constructor to use for connecting to the backend socket. 

Each method defined in the class takes a set of input parameters defined for that specific request. A python dictionary is then formatted, converted into a json string and then sent to the backend through the defined "__send_data" method.

### Example method, run_batch

The run_batch method defined:

```py
# Sends a request to the backend to start a batch job
def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
    data = {
        "METHOD": "run-batch",
        "model": model,
        "dataset": dataset,
        "name": name
    }
    if inj_params:
        data["inj_params"] = inj_params

    self.__send_data(data, response=False)
```
Where "inj_params" is a dictionary defined as: 
```py
inj_params = {
    "anomaly_type": anomaly_type,
    "timestamp": timestamp,
    "magnitude": magnitude,
    "percentage": percentage,
    "duration": duration,
    "columns": columns
}
```

### __send_data method

The __send_data function initiates a socket object, connects to the backend engine and then sends the json string through the socket. The client will wait for a response from the backend unless the response parameter is set to False. If method is "import-dataset" two messages is sent, one contains the request dictionary and one contains the file contents of the dataset. 

```py
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
            # Make sure the recieving end reads two separate messages
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
```

### All method declarations for BackendAPI class

For the purposes of defining what the BackendAPI provides, here is a list of all instance methods. These are created to facilitate all the interaction needed from a users perspective and is used by the Fronted as well as able to be used from the CLI-tool. 

```py
def __init__(self, host: str, port: int) -> None:
def run_batch(self, model: str, dataset: str, name: str, inj_params: dict=None) -> None:
def run_stream(self, model: str,dataset: str, name: str, speedup: int, inj_params: dict=None) -> None:
def change_model(self, model: str, name: str) -> None:
def change_method(self, injection_method: str, name: str) -> None:
def get_data(self, timestamp: str, name: str) -> str:
def inject_anomaly(self, timestamps: list[int], name: str) -> None:
def get_running(self) -> str:
def cancel_job(self, name: str) -> None:
def get_models(self) -> str:
def get_injection_methods(self) -> str:
def get_datasets(self) -> str:
def get_all_jobs(self) -> str:
def import_dataset(self, file_path: str, timestamp_column: str) -> None:
```