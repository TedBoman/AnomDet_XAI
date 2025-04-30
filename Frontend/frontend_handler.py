import sys
import os
import traceback
import pandas as pd
import json
from io import StringIO

from api import BackendAPI

class FrontendHandler:
    def __init__(self, host, port):
        self.api = BackendAPI(host, port)

    def check_name(self, job_name) -> str:
        response = self.handle_get_all_jobs()
        if job_name in response:
            return "name-error"
        return "success"
    
    def handle_run_batch(self, selected_dataset, selected_model, job_name, inj_params: dict=None, label_column=None, xai_params=None, model_params=None) -> str:
        response = self.check_name(job_name)

        if response == "success":
            if inj_params is None:
                self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=None, label_column=label_column, xai_params=xai_params, model_params=None)
            else:
                self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=[inj_params], label_column=label_column, xai_params=xai_params, model_params=None)

        return response

    def handle_run_stream(self, selected_dataset, selected_model, job_name, speedup, inj_params: dict=None, label_column=None, xai_params=None, model_params=None) -> str:
        response = self.check_name(job_name)

        if response == "success":
            if inj_params is None:
                response = self.api.run_stream(selected_model, selected_dataset, job_name, speedup, inj_params=None, label_column=label_column, xai_params=xai_params, model_params=None)
            else:
                self.api.run_stream(selected_model, selected_dataset, job_name, speedup, inj_params=[inj_params], label_column=label_column, xai_params=xai_params, model_params=None)

        return response

    def handle_get_data(self, timestamp, job_name):
        data = self.api.get_data(timestamp, job_name)
        df = pd.read_json(StringIO(data["data"]), orient="split")
        return df
        
    def handle_get_running(self):
        return self.api.get_running()

    def handle_cancel_job(self, job_name):
        response = self.check_name(job_name)

        if response == "name-error":
           self.api.cancel_job(job_name)

        return response

    def handle_get_models(self):
        models = json.loads(self.api.get_models())
        return models["models"]
    
    def handle_get_xai_methods(self):
        methods_list = [] # Default
        try:
            response_data = self.api.get_xai_methods() # Call backend

            # Check for connection failure / empty response
            if response_data is None or response_data == "":
                 print("Warning: API get_xai_methods returned None or empty.")
                 return methods_list # Return empty list

            # Try parsing JSON
            methods_data = json.loads(response_data)

            # Check expected structure
            if isinstance(methods_data, dict) and 'methods' in methods_data and isinstance(methods_data['methods'], list):
                 methods_list = methods_data["methods"]
            else:
                 # Log unexpected structure (could be an error JSON from backend)
                 print(f"Warning: Unexpected JSON structure from get_xai_methods: {methods_data}")
                 # Keep methods_list empty

        except json.JSONDecodeError as json_err:
            print(f"Error decoding JSON from get_xai_methods: {json_err} | Data: {response_data!r}")
        except ConnectionError as conn_err: # Catch connection errors if api raises them
            print(f"Connection Error in get_xai_methods: {conn_err}")
        except Exception as e:
            print(f"Generic Error in handle_get_xai_methods: {e}")
            traceback.print_exc()

        return methods_list # Always return a list

    def handle_get_injection_methods(self):
        injection_methods = json.loads(self.api.get_injection_methods())
        return injection_methods["injection_methods"]

    def handle_get_datasets(self):
        datasets = json.loads(self.api.get_datasets())
        return datasets["datasets"]

    def handle_import_dataset(self, file_path, timestamp_column: str):
        self.api.import_dataset(file_path, timestamp_column)

    def handle_get_all_jobs(self):
        jobs = json.loads(self.api.get_all_jobs())
        return jobs["jobs"]

    def handle_get_columns(self, job_name):
        response = self.check_name(job_name)

        if response == "name-error":
            columns = json.loads(self.api.get_columns(job_name))
            return columns["columns"]

        return response

    def handle_get_dataset_columns(self, dataset):
        if dataset == None:
            return []
        response = self.api.get_dataset_columns(dataset)
        columns = json.loads(response)
        return columns["columns"]