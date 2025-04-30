import socket
import json
import sys
import threading
from time import sleep
import os
import execute_calls
from timescaledb_api import TimescaleDBAPI
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
from Simulator.FileFormats.read_csv import read_csv

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
GRAFANA_URL = f"http://{os.getenv('GRAFANA_HOST')}:{os.getenv('GRAFANA_PORT')}"
GRAFANA_API_KEY = os.getenv('GRAFANA_API_KEY')

DATASET_DIRECTORY = "./Datasets/"

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
            label_column = data.get("label_column", None)

            inj_params = data.get("inj_params", None)
            xai_params = data.get("xai_params", None)
            model_params = data.get("model_params", None)

            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }
            
            new_thread = threading.Thread(
            target=execute_calls.run_batch,
                args=(
                    db_conn_params,
                    model,
                    dataset_path,
                    name,
                    inj_params,
                    debug,
                    label_column,
                    xai_params,
                    model_params,
                )
            )
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
            label_column = data.get("label_column", None)

            inj_params = data.get("inj_params", None)
            xai_params = data.get("xai_params", None)
            model_params = data.get("model_params", None)
            
            db_conn_params = {
                "user": DATABASE["USER"],
                "password": DATABASE["PASSWORD"],
                "host": DATABASE["HOST"],
                "port": DATABASE["PORT"],
                "database": DATABASE["DATABASE"]
            }

            stream_thread = threading.Thread( # Renamed variable for clarity
                target=execute_calls.run_stream,
                args=(
                    db_conn_params,
                    model,
                    dataset_path,
                    name,
                    speedup,
                    inj_params,
                    debug,
                    label_column, # Pass label_column variable
                    xai_params,    # Pass xai_params variable
                    model_params
                )
            )            

            stream_thread.daemon = True
            stream_thread.start()
            detection_thread = threading.Thread(target=execute_calls.single_point_detection, args=(backend_data["db_api"], new_thread, model, name, dataset_path))
            detection_thread.daemon = True
            detection_thread.start()

            job = {
                "name": name,
                "type": "stream",
                "thread": stream_thread
            }

            backend_data["started-jobs"].append(job)
            
        case "get-data":
            if data["to_timestamp"] == None:
                df = backend_data["db_api"].read_data(datetime.fromtimestamp(int(data["from_timestamp"])), data["job_name"])
            else:
                df = backend_data["db_api"].read_data(datetime.fromtimestamp(int(data["from_timestamp"])), data["job_name"], datetime.fromtimestamp(int(data["to_timestamp"])))
            df["timestamp"] = df["timestamp"].apply(execute_calls.map_to_timestamp)
            df["timestamp"] = df["timestamp"].astype(float)
            data_json = df.to_json(orient="split")

            df_dict = {
                "data": data_json
            }
            df_json = json.dumps(df_dict)
            conn.sendall(bytes(df_json, encoding="utf-8"))
            print("Data sent")
        case "get-running":
            jobs = []
            for job in backend_data["running-jobs"]:
                new_job = {
                    "name": job["name"],
                    "type": job["type"]
                }
                jobs.append(new_job)
            running_dict = {
                "running": jobs
            }
            running_json = json.dumps(running_dict)
            conn.sendall(bytes(running_json, encoding="utf-8"))
        case "cancel-job":
            __cancel_job(data["job_name"])
        case "get-models":
            models = execute_calls.get_models()
            models_dict = {
                                "models": models
                            }
            models_json = json.dumps(models_dict)
            conn.sendall(bytes(models_json, encoding="utf-8"))
        case "get-xai-methods":
            methods = execute_calls.get_xai_methods()
            print(f"sending columns: {methods}")
            methods_dict = {
                                "methods": methods
                            }
            methods_json = json.dumps(methods_dict)
            conn.sendall(bytes(methods_json, encoding="utf-8"))
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
            columns = backend_data["db_api"].get_columns(data["name"])
            columns_dict = {
                                "columns": columns
                            }
            columns_json = json.dumps(columns_dict)
            conn.sendall(bytes(columns_json, encoding="utf-8"))
        case "get-dataset-columns":
            file_reader = read_csv(DATASET_DIRECTORY + data["dataset"])
            columns = file_reader.get_columns()
            print(f"sending columns: {columns}")
            columns_dict = {
                                "columns": columns
                            }
            columns_json = json.dumps(columns_dict)
            conn.sendall(bytes(columns_json, encoding="utf-8"))
        case _: 
            response_json = json.dumps({"error": "method-error-response" })
            conn.sendall(bytes(response_json, encoding="utf-8"))      
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()
            
def __cancel_job(job_name: str) -> None:
    print("Cancelling job...")
    for job in backend_data["running-jobs"]:
        if job["name"] == job_name:
            #if job["type"] == "stream" and job["thread"].is_alive():
                # Stop the streaming thread if it's a running stream job
            backend_data["db_api"].drop_table(job_name)
            backend_data["running-jobs"].remove(job)
            break

if __name__ == "__main__": 
    main()