import socket
import json
import threading
from time import sleep
import os
import execute_calls
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('HOST')
PORT = int(os.getenv('PORT'))

DATASET_DIRECTORY = "../Datasets/"

backend_data = {
    "batch-jobs": [],
    "stream-jobs": [],
    "running-models": []
}

def main():
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
        except Exception as e:
            pass

# Handles the incoming requests and sends a response back to the client
def __handle_api_call(conn, data: dict) -> None:
    match data["METHOD"]:
        case "run-batch":
            print("test")
            model = data["model"]
            injection_method = data["injection_method"]
            dataset_path = DATASET_DIRECTORY + data["dataset"]

            df = pd.read_csv(dataset_path, low_memory=False)
            df["is_anomaly"] = False
            df["injected_anomaly"] = False
            print(df)
            #result = execute_calls.run_batch(model, injection_method, path)
            print(df)
            test_json = json.dumps({"test": "run-batch-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "run-stream":
            test_json = json.dumps({"test": "run-stream-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "change-model":
            test_json = json.dumps({"test": "change-model-respons" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "change-method":
            test_json = json.dumps({"test": "change-method-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "get-data":
            test_json = json.dumps({"test": "get-data-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "inject-anomaly":
            test_json = json.dumps({"test": "inject-anomaly-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "get-running":
            test_json = json.dumps({"test": "get-running-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "cancel":
            
            test_json = json.dumps({"test": "cancel-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case "get-models":
            models = execute_calls.get_models()
            models_dict = {
                                "models": models
                            }
            models_json = json.dumps(models_dict)
            conn.sendall(bytes(models_json, encoding="utf-8"))
        case "get-injection-methods":
            injection_methods = execute_calls.get_injection_methods()
            injection_methods_dict = {
                                "injection_methods": injection_methods
                            }
            injection_methods_json = json.dumps(injection_methods_dict)
            conn.sendall(bytes(injection_methods_json, encoding="utf-8"))
        case "get-datasets":
            datasets = execute_calls.get_datasets()
            datasets_dict = {
                                "datasets": datasets
                            }
            datasets_json = json.dumps(datasets_dict)
            conn.sendall(bytes(datasets_json, encoding="utf-8"))
        case "upload-dataset":
            test_json = json.dumps({"test": "upload-dataset-response" })
            conn.sendall(bytes(test_json, encoding="utf-8"))
        case _: 
            response_json = json.dumps({"error": "method-error-response" })
            conn.sendall(bytes(response_json, encoding="utf-8"))        

if __name__ == "__main__": 
    main()