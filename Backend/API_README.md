# Backend API and CLI-tool documentation

## CLI-tool

We've made it possible to send requests to the backend with the backend api through the command-line. This is made possible by the main-function in "backend_api.py". The main function parses the command-line arguments and calls, handles errors and calls appropiate function sending data to the backend. Run "python backend_api.py help" for CLI-tool documentation.

## BackendAPI

Class designed to serve all requests/responses to and from the backend. It is initialized with a host and port sent to the constructor to use for connecting to the backend socket. 

Each method defined in the class takes a set of input parameters defined for that specific request. A python dictionary is then formatted, converted into a json string and then sent to the backend through the defined "__send_data" method.

### Example method, run_batch

The run_batch method defined:

```py
# Sends a request to the backend to start a batch job
def run_batch(self, model: str, injection_method: str, dataset: str, name: str) -> str:
    data = {
        "METHOD": "run-batch",
        "model": model,
        "injection_method": injection_method,
        "dataset": dataset
        "name": name
    }
    return self.__send_data(json.dumps(data))
```

### __send_data method

The __send_data function initiates a socket object, connects to the backend engine and then sends the json string through the socket. Current implementation always waits for a response from the backend.

```py
# Initates connection to backend and sends json data through the socket
def __send_data(self, data: str) -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.host))
        sock.sendall(bytes(data, encoding="utf-8"))
        data = sock.recv(1024)
    except Exception as e:
        print(e)

    if data:
        return json.loads(data)
```

### All method declarations for BackendAPI class

For the purposes of defining what the BackendAPI provides, here is a list of all instance methods. These are created to facilitate all the interaction 

```py
def __init__(self, host: str, port: int) -> None:
def run_batch(self, model: str, injection_method: str, dataset: str) -> str:
def run_stream(self, model: str, injection_method: str, dataset: str, name: str) -> str:
def change_model(self, model: str, name: str) -> str:
def change_method(self, injection_method: str, name: str) -> str:
def get_data(self, timestamp: str, name: str) -> str:
def inject_anomaly(self, timestamps: list[int], name: str) -> str:
def get_running(self) -> str:
def cancel_job(self, name: str) -> str:
def get_models(self) -> str:
def get_injection_methods(self) -> str:
def get_datasets(self) -> str:
def import_dataset(self, file_path: str) -> str:
```