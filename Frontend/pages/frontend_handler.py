from Backend.backend_api import BackendAPI


def user_request(selected_dataset, selected_model, selected_inj_method, job_type):
    match job_type:
        case "batch":
            BackendAPI().run_batch(selected_model, selected_inj_method, selected_dataset)
            return
        case "stream":
            BackendAPI().run_stream(selected_model, selected_inj_method, selected_dataset)
            return
        
def send_socket_request(data):
    pass

def get_models():
    pass

def get_datasets():
    pass


    
