import sys
import os

from api import BackendAPI

class FrontendHandler:
    def __init(self, host, port):
        self.api = BackendAPI(host, port)
    
    def handle_run_batch(self, selected_dataset, selected_model, job_name, inj_params: dict=None) -> str:
        response = self.handle_get_all_jobs()
        if job_name in response["jobs"]:
            return "name-error"

        self.api.run_batch(selected_model, selected_dataset, job_name, inj_params=inj_params)
        return "success"

    def handle_run_stream(self, selected_dataset, selected_model, job_name, inj_params: dict=None) -> str:
        response = self.handle_get_all_jobs()
        if job_name in response["jobs"]:
            return "name-error"

        self.api.run_stream(selected_model, selected_dataset, job_name, inj_params=inj_params)
        return "success"

    def handle_change_model(self, selected_model, job_name):
        return self.api.change_model(selected_model, job_name)

    def handle_change_method(self, selected_injection_method, job_name):
        return self.api.change_method(selected_injection_method, job_name)

    def handle_get_data(self, timestamp, job_name):
        return self.api.get_data(timestamp, job_name)
        
    def handle_get_running(self):
        return self.api.get_running()

    def handle_cancel_job(self, job_name):
        self.api.cancel_job(job_name)

    def handle_get_models(self):
        return self.api.get_models()

    def handle_get_injection_methods(self):
        return self.api.get_injection_methods()

    def handle_get_datasets(self):
        return self.api.get_datasets()

    def handle_import_dataset(self, file_path, timestamp_column=None):
        self.api.import_dataset(file_path, timestamp_column)

    def handle_get_all_jobs(self):
        self.api.get_all_jobs()

    def handle_get_columns(self, job_name):
        return self.api.get_columns(job_name)
