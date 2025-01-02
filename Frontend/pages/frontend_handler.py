import sys
import os
from dotenv import load_dotenv
# Backend içindeki api klasörüne giden yolu ekleyin
backend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Backend', 'api')
)
sys.path.append(backend_path)

# Artık Backend/api içindeki api.py dosyasından BackendAPI'yi import edebiliriz
from api import BackendAPI

class FrontendHandler:
    def __init(self, host, port):
        self.api = BackendAPI(host, port)

    def user_request(selected_dataset, selected_model, job_type, selected_injection_method=None, range=None):
        load_dotenv()
        HOST = 'Backend'
        PORT = int(os.getenv('BACKEND_PORT'))
        api = BackendAPI(HOST, PORT)
        print(selected_dataset, selected_model, job_type, selected_injection_method, range)
        match job_type:
            case "batch":
                if selected_injection_method:
                    api.run_batch(selected_model, selected_dataset, "name", selected_injection_method)
                else: 
                    api.run_batch(selected_model, selected_dataset, "name")
                return
            case "stream":
                if selected_injection_method:
                    api.run_stream(selected_model, selected_dataset, "name", range, selected_injection_method)
                else:
                    api.run_stream(selected_model, selected_dataset, "name")
                return
            
    def handle_change_model(self, selected_model, job_name=None):
        return self.api.change_model(selected_model, job_name)
        

    def handle_change_method(self, selected_injection_method, job_name=None):
        return self.api.change_method(selected_injection_method, job_name)

    def handle_get_data(self, timestamp, job_name):
        return self.api.get_data(timestamp, job_name)
        
    def handle_get_running(self):
        return self.api.get_running()

    def handle_cancel_job(self, job_name=None):
        return self.api.cancel_job(job_name)

    def handle_get_models(self):
        return self.api.get_models()

    def handle_get_injection_methods(self):
        return self.api.get_injection_methods()

    def handle_get_datasets(self):
        return self.api.get_datasets()

    def handle_import_dataset(self, file_path, timestamp_column=None):
        return self.api.import_dataset(file_path, timestamp_column)

    def handle_get_all_jobs(self):
        return self.api.get_all_jobs()

    def handle_get_columns(self, job_name):
        return self.api.get_columns(job_name)

    def initiate_connection(self, data, response):
        #Think we should initiate the connection to the backend here when the frontend is sorted out
        pass


        



    

   



    
