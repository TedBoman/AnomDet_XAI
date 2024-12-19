from api import BackendAPI


def user_request(selected_dataset, selected_model, job_type):
    print(selected_dataset, selected_model, job_type)
    match job_type:
        case "batch":
            BackendAPI().run_batch(selected_model, None, selected_dataset)
            return
        case "stream":
            BackendAPI().run_stream(selected_model, None, selected_dataset)
            return
        
#@MaxStrang, here implement backend api defined functions.


    
