import sys
from os import path

DOC = """Invoke backend API from command line with the following commands:

    1. "python backend_api.py run-batch <model> <injection-methd> <path-to-batch-file>"
        starts anomaly detection of batch data from the given file with the given model and injection method

    2. "python backend_api.py run-stream <path-to-stream-file>" starts anomaly detection of stream data from the given file
        starts anomaly detection of stream data from the given file with the given model and injection method

    3. "python backend_api.py change-model <model> <dataset-running>"
        changes the model used for anomaly detection for the currently run batch or stream named <dataset-running> to <model>

    4. "python backend_api.py change-injectio <injection-method> <dataset-running>"
        changes the injection method used for anomaly detection for the currently run batch or stream named <dataset-running> to <injection-method>

    5. "python backend_api.py cancel <dataset-running>" prints this help message"
        cancels the currently running batch or stream named <dataset-running>

    6. "python backend_api.py get-data <dataset-running>"
        get all processed data from <dataset-running>, meaning just the data that has gone through our detection model
    
    7. "python backend_api.py help"
        prints this help message"""

def main(argv: list[str]):
    # Start a batch job in the backend if the command is "run-batch"
    if argv[1] == "run-batch":
        if len(argv) != 5:
            handle_error(1, "Invalid number of arguments")
        run_batch(argv[2], argv[3], argv[4])

    # Start a stream job in the backend if the command is "run-stream"
    elif argv[1] == "run-stream":
        if len(argv) != 5:
            handle_error(1, "Invalid number of arguments")
        run_stream(argv[2], argv[3], argv[4])
        
    # Change the model used for a running job if the command is "change-model"
    elif argv[1] == "change-model":
        if (len(argv) != 4):
            handle_error(1, "Invalid number of arguments")
        change_model(argv[2], argv[3])

    # Change the injection method used for a running job if the command is "change-injection"
    elif argv[1] == "change-injection":
        if (len(argv) != 4):
            handle_error(1, "Invalid number of arguments")
        change_method(argv[2], argv[3])
    
    # Get data from a running job if the command is "get-data", the backend will return data that has gone through the detection model
    elif argv[1] == "get-data":
        if (len(argv) != 3):
            handle_error(1, "Invalid number of arguments")
        get_data(argv[2])

    # Print information about the backend API command line tool if the command is "help"
    elif argv[1] == "help":
        print(DOC)

    # Print an error message if the command is not recognized
    else: 
        handle_error(3, f'argument "{argv[1]}" not recognized as a valid command')
        

def run_batch(model: str, injection_method: str, file_path: str):
    if not path.isfile(path):
        handle_error(2, "File not found")
    pass

def run_stream(model: str, injection_method: str, file_path: str):
    if not path.isfile(path):
        handle_error(2, "File not found")
    pass

def change_model(model: str, name: str):
    pass

def change_method(method: str, name: str):
    pass

def get_data(name: str):
    pass

def handle_error(code: int, message: str):
    print(message)
    exit(code)

if __name__ == "__main__":
    main(sys.argv)