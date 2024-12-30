import socket
import json
import threading
from time import sleep
import os
import execute_calls
import pandas as pd
from timescaledb_api import TimescaleDBAPI
from dotenv import load_dotenv

load_dotenv()
BACKEND_HOST = os.getenv('BACKEND_HOST')
BACKEND_PORT = int(os.getenv('BACKEND_PORT'))
DATABASE = {
    "HOST": os.getenv('DATABASE_HOST'),
    "PORT": int(os.getenv('DATABASE_PORT')),
    "USER": os.getenv('DATABASE_USER'),
    "PASSWORD": os.getenv('DATABASE_PASSWORD'),
    "DATABASE": os.getenv('DATABASE_NAME')
}
DATABASE_PORT = os.getenv('DATABASE_HOST')

DATASET_DIRECTORY = "./Datasets/"

# 
backend_data = {
    "started-jobs": [],
    "running-jobs": []
}

def main():
    # Start a thread listening for requests
    listener_thread = threading.Thread(target=__request_listener)
    listener_thread.daemon = True
    listener_thread.start()

    db_conn_params = {
        "user": DATABASE["USER"],
        "password": DATABASE["PASSWORD"],
        "host": DATABASE["HOST"],
        "port": DATABASE["PORT"],
        "database": DATABASE["DATABASE"]
    }

    backend_data["db_api"] = TimescaleDBAPI(db_conn_params)

    print("Main thread started...")
    # Main loop serving the backend logic
    try:
        while True:
            for job in backend_data["started-jobs"]:
                # If the job is a batch job, check if the job is finished to move it to the running-jobs list
                if job["type"] == "batch":
                    if job["thread"].is_alive() == False:
                        backend_data["running-jobs"].append(job)
                        backend_data["started-jobs"].remove(job)
                # If the job is a stream job, check if there is a table with the name of the job in the database to add to run job
                else:
                    found = backend_data["db_api"].table_exists(job["name"])
                    if found:
                        backend_data["running-jobs"].append(job)
                        backend_data["started-jobs"].remove(job)
                sleep(1)
    except KeyboardInterrupt:
        print("Exiting backend...")

# Listens for incoming requests and handles them through the __handle_api_call function
def __request_listener():
    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((BACKEND_HOST, BACKEND_PORT))
        sock.listen()
        sock.settimeout(1)
    except Exception as e:
        print(e)

    while True:
        try: 
            conn, addr = sock.accept()
            print(f'Connected to {addr}')
            recv_data = conn.recv(1024)
            recv_data = recv_data.decode("utf-8")
            recv_dict = json.loads(recv_data)
            print(f"Received request: {recv_dict}")
            __handle_api_call(conn, recv_dict)
        except Exception as e:
            if str(e) != "timed out":
                print(e)
            pass

# Handles the incoming requests and sends a response back to the client
def __handle_api_call(conn, data: dict) -> None:
    match data["METHOD"]:
        case "run-batch":
            model = data["model"]
            dataset_path = DATASET_DIRECTORY + data["dataset"]
            name = data["name"]
            debug = data["debug"]

            print(data)

            inj_params = data.get("inj_params", None)

            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }
            
            new_thread = threading.Thread(target=execute_calls.run_batch, args=(db_conn_params, model, dataset_path, name, inj_params, debug))
            new_thread.daemon = True
            new_thread.start()

            job = {
                "name": name,
                "type": "batch",
                "thread": new_thread
            }

            backend_data["started-jobs"].append(job)
            
        case "run-stream":
            model = data["model"]
            dataset_path = DATASET_DIRECTORY + data["dataset"]
            name = data["name"]
            speedup = data["speedup"]
            debug = data["debug"]
            
            inj_params = data.get("inj_params", None)
            
            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }

            new_thread = threading.Thread(target=execute_calls.run_stream, args=(db_conn_params, model, dataset_path, name, speedup, inj_params, debug))
            new_thread.daemon = True
            new_thread.start()

            job = {
                "name": name,
                "type": "stream",
                "thread": new_thread
            }

            backend_data["started-jobs"].append(job)

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
            running_dict = {
                                "running": backend_data["running-jobs"]
                            }
            running_json = json.dumps(running_dict)
            conn.sendall(bytes(running_json, encoding="utf-8"))
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
        case "import-dataset":
            path = DATASET_DIRECTORY + data["name"]
            conn.settimeout(1)
            # If the file does not exist, read the file contents written to the socket
            if not os.path.isfile(path):
                execute_calls.import_dataset(conn, path, data["timestamp_column"])
            # If the file already exists, empty the socket buffer and do nothing
            else:
                data = conn.recv(1024)
                while data:
                    data = conn.recv(1024)
        case "get-all-jobs":
            job_names = []

            for job in backend_data["running-jobs"]:
                job_names.append(job["name"])
            
            for job in backend_data["started-jobs"]:
                job_names.append(job["name"])
            
            jobs_dict = {
                            "jobs": job_names
                        }
            jobs_json = json.dumps(jobs_dict)
            conn.sendall(bytes(jobs_json, encoding="utf-8"))
        case "get-columns":
            columns = execute_calls.get_columns(data["name"])
            columns_dict = {
                                "columns": columns
                            }
            columns_json = json.dumps(columns_dict)
            conn.sendall(bytes(columns_json, encoding="utf-8"))
        case _: 
            response_json = json.dumps({"error": "method-error-response" })
            conn.sendall(bytes(response_json, encoding="utf-8"))        

if __name__ == "__main__": 
    main()