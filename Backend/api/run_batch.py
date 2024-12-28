import json
from api import BackendAPI

def run_batch(api: BackendAPI) -> dict:
    # Gather available models for detection
    response = api.get_models()
    models = json.loads(response)["models"]
    print(f"Theses are the models provided by AnomDet: {models}")
    model = input("Enter the model to use: ")
    while model not in models:
        model = input("Model not found, please enter a valid model: ")

    # Gather available datasets for detection
    response = api.get_datasets()
    datasets = json.loads(response)["datasets"]
    print(f"Theses are the datasets provided by AnomDet: {datasets}")
    dataset = input("Enter the dataset to use: ")
    while dataset not in datasets:
        dataset = input("Dataset not found, please enter a valid dataset: ")

    # Gather available jobs to make sure user gives a unique name
    response = api.get_all_jobs()
    jobs = json.loads(response)["jobs"]
    name = input("Enter what you want to name the job: ")
    if name in jobs:
        name = input("Name already in use, please enter a new name: ")
        
    # Ask user if they want to inject an anomalies
    insert_anomaly = input("Do you want to insert anomalies? (y/n): ")

    if insert_anomaly == "y":
        # Gather injection methods for anomalies
        response = api.get_injection_methods()
        injection_methods = json.loads(response)["injection_methods"]
        print(f"Theses are the injection methods provided by AnomDet: {injection_methods}")
        injection_method = input("Enter injection method: ")
        while injection_method not in injection_methods:
            injection_method = input("Injection method not found, please enter a valid anomaly type: ")

        timestamp = input("Enter the timestamp to start anomaly: ")
        magnitude = input("Enter the magnitude of the anomaly: ")
        percentage = input("Enter the percentage of data: ")
        duration = input("Enter the duration of the anomalies: ")
        columns_string = input("Enter the columns to inject anomalies into, as a comma separated list (a,b,c,d,...): ")
        inj_params = {
            "anomaly_type": injection_method,
            "timestamp": timestamp,
            "magnitude": magnitude,
            "percentage": percentage,
            "duration": duration,
            "columns": columns_string.split(',')
        }
        print(inj_params)
        api.run_batch(model, dataset, name, inj_params)
    else:
        api.run_batch(model, dataset, name)