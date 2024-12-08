import socket
import json
import threading
from time import sleep
import os
from pathlib import PureWindowsPath, PurePosixPath
import platform
import ML_models

HOST = "localhost"
PORT = 9524

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./injection_methods"

backend_data = {
    "batch-jobs": [],
    "stream-jobs": [],
    "models": [],
    "injection-methods": []
}

def main():
    # Initialize the backend with some models and injection methods
    backend_data["models"] = __get_models()
    backend_data["injection-methods"] = __get_injection_methods()

    print(f"Models: {backend_data['models']}")
    print(f"Injection methods: {backend_data['injection-methods']}")

    # Start a thread listening for requests
    listener_thread = threading.Thread(target=__request_listener)
    listener_thread.daemon = True
    listener_thread.start()

    i = 1
    print("Counting in main thread...")
    # Main loop serving the backend logic
    try:
        while True:
            print(i)
            i += 1
            sleep(1)
    except KeyboardInterrupt:
        print("Exiting backend...")

# Returns a list of models implemented in MODEL_DIRECTORY
def __get_models() -> list:
    models = []
    for path in os.listdir(MODEL_DIRECTORY):
        if os.path.isfile(os.path.join(MODEL_DIRECTORY, path)):
            model_name = path.split(".")[0]
            models.append(model_name)

    models.remove("model_interface")
    models.remove("__init__")
    models.remove("setup")
    
    return models

def __get_injection_methods() -> list:
    injection_methods = ["not implemented"]
    '''
    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)
    '''
    return injection_methods
    
# Listens for incoming requests and handles them through the __handle_api_call function
def __request_listener():
    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((HOST, PORT))
        sock.listen()
        sock.settimeout(1)
    except Exception as e:
        print(e)

    while True:
        try: 
            conn, addr = sock.accept()
            print(f'Connected to {addr}')
            recv_data = conn.recv(1024)
            recv_dict = json.loads(recv_data)
            __handle_api_call(conn, recv_dict)
            print(f"Received request: {recv_dict}")
            conn.close()
        except Exception as e:
            pass
    

# Handles the incoming requests and sends a response back to the client
def __handle_api_call(conn, data: dict) -> None:
    if data["METHOD"] == "run-batch":
        if platform.system() == "Windows":              # Split the path apart to get filename
            path = PureWindowsPath(data["file_path"])
        else:
            path = PurePosixPath(data["file_path"])
        file_name = path.parts[-1].split(".")[0]
        test_json = json.dumps({"test": f'running batch job on {file_name} with model {data["model"]} and injection method {data["injection_method"]}' })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "run-stream":
        test_json = json.dumps({"test": "run-stream-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "change-model":
        test_json = json.dumps({"test": "change-model-respons" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "change-method":
        test_json = json.dumps({"test": "change-method-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-data":
        test_json = json.dumps({"test": "get-data-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "inject-anomaly":
        test_json = json.dumps({"test": "inject-anomaly-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-running":
        test_json = json.dumps({"test": "get-running-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "cancel":
        test_json = json.dumps({"test": "cancel-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-models":
        test_json = json.dumps({"models": backend_data["models"] })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-injection-methods":
        test_json = json.dumps({"test": "get-injection-methods-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    else: 
        test_json = json.dumps({"test": "error-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))        

if __name__ == "__main__": 
    main()