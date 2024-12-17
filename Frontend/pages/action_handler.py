


def user_request(selected_dataset, selected_model, selected_inj_method):
    match selected_inj_method:
        case None:
            BckendAPI.get_dataset(selected_dataset)
            BckendAPI.get_model(selected_model)
        case _:
            BckendAPI.get_dataset(selected_dataset)
            BckendAPI.get_model(selected_model)
            BckendAPI.get_injection_method(selected_inj_method)
    