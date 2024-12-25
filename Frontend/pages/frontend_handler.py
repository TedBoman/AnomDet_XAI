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
        



    
