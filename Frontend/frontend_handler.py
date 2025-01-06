import sys
import os
import pandas as pd
import json

from api import BackendAPI

class FrontendHandler:
    def __init__(self, host, port):
        self.api = BackendAPI(host, port)

    def check_name(self, job_name) -> str:
        response = self.handle_get_all_jobs()
        if job_name in response:
            return "name-error"
        return "success"
    
    def handle_run_batch(self, selected_dataset, selected_model, job_name, inj_params: dict=None) -> str:
        response = self.check_name(job_name)

        if response == "success":
            if inj_params is None:
                self.api.run_batch(selected_model, selected_dataset, job_name)
            else:
                self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=inj_params)

        return response

    def handle_run_stream(self, selected_dataset, selected_model, job_name, speedup, inj_params: dict=None) -> str:
        response = self.check_name(job_name)

        if response == "success":
            if inj_params is None:
                response = self.api.run_stream(selected_model, selected_dataset, job_name, speedup)
            else:
                self.api.run_stream(selected_model, selected_dataset, job_name, speedup, inj_params=inj_params)

        return response

    def handle_change_model(self, selected_model, job_name):
        return self.api.change_model(selected_model, job_name)

    def handle_change_method(self, selected_injection_method, job_name):
        return self.api.change_method(selected_injection_method, job_name)

    def handle_get_data(self, timestamp, job_name):
        data = self.api.get_data(timestamp, job_name)
        print(data)
        df = data["data"]
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

        if response == "success":
            columns = json.loads(self.api.get_columns(job_name))
            return columns["columns"]

        return response

    def handle_get_dataset_columns(self, dataset):
        if dataset == None:
            return []
        response = self.api.get_dataset_columns(dataset)
        columns = json.loads(response)
        columns["columns"].remove("timestamp")
        return columns["columns"]