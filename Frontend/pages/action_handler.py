from Backend.backend_api import BackendAPI


def job_request(selected_dataset, selected_model, selected_inj_method, job_type):
    match job_type:
        case 'batch':
            BackendAPI.run_batch_job(selected_dataset, selected_model, selected_inj_method)
            pass
        case 'stream':
            BackendAPI.run_stream_job(selected_dataset, selected_model, selected_inj_method)
            pass
    
