from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update
import json
from get_handler import get_handler
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN

def get_index_callbacks(app):
    @app.callback(
        Output("injection", "style"),
        Input("injection-check", "value")
    )
    def update_injection_panel(selected):
        if "use_injection" in selected:
            return {"display": "block"}
        return {"display": "none"}
    @app.callback(
        Output("column-dropdown", "options"),
        Input("dataset-dropdown", "value")
    )
    def update_column_dropdown(selected_dataset):
        handler = get_handler()
        columns = handler.handle_get_dataset_columns(selected_dataset)
        return [{"label": col, "value": col} for col in columns]

    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-jobs-list", "children")
    )
    def toggle_active_jobs_section(children):
        # Show the active jobs section if there are active jobs
        if children == "No active jobs found.":
            return {"display": "none"}
        return {"display": "block", "marginTop": "30px"}
        
    # Callback to display confirmation box
    @app.callback(
        Output({"type": "confirm-box", "index": MATCH}, "displayed"),
        Input({"type": "remove-dataset-btn", "index": MATCH}, "n_clicks")
    )
    def display_confirm(value):
        return True if value else False

    # Callback to manage active jobs
    @app.callback(
        [Output("active-jobs-list", "children")],
        [Input("job-interval", "n_intervals"), Input({"type": "confirm-box", "index": ALL}, "submit_n_clicks")],
        [State("active-jobs-json", "data")],
        
    )
    def manage_and_remove_active_jobs(children, submit_n_clicks, active_jobs_json):
        ctx = callback_context

        triggered_id = ctx.triggered_id

        if triggered_id == None:
            return no_update

        handler = get_handler()

        if triggered_id != "job-interval":
            job = triggered_id["index"]
            handler.handle_cancel_job(job)

        active_jobs = json.loads(handler.handle_get_running())
        active_jobs = active_jobs["running"]
        jobs_json = json.dumps(active_jobs)

        if jobs_json == active_jobs_json:
            return no_update
        
        return create_active_jobs(active_jobs)

    @app.callback(
            [Output("popup", "style"), Output("popup-interval", "disabled"), Output("popup", "children")],
            [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
            [
                State("dataset-dropdown", "value"),
                State("detection-model-dropdown", "value"),
                State("mode-selection", "value"),
                State("name-input", "value"),
                State("injection-method-dropdown", "value"),
                State("timestamp-input", "value"),
                State("magnitude-input", "value"),
                State("percentage-input", "value"),
                State("duration-input", "value"),
                State("column-dropdown", "value"),
                State("injection-check", "value"),
                State("speedup-input", "value"),
                State("popup", "style")
            ]
            )
    def start_job_handler(
                            n_clicks,
                            n_intervals,
                            selected_dataset,
                            selected_detection_model,
                            selected_mode,
                            job_name,
                            selected_injection_method,
                            timestamp,
                            magnitude,
                            percentage,
                            duration,
                            columns,
                            inj_check,
                            speedup,
                            style
                        ):   
        handler = get_handler()
        children = "Job has started!"

        ctx = callback_context
        if not ctx.triggered:
            return style, True, children

        trigger = ctx.triggered[0]["prop_id"]
        if trigger == "start-job-btn.n_clicks":
            response = handler.check_name(job_name)
            if job_name == "":
                style.update({"backgroundColor": "#e74c3c"})
                children = "Job name cannot be empty!"
            elif response == "success":
                if "use_injection" in inj_check:
                    inj_params = {
                                    "anomaly_type": selected_injection_method,
                                    "timestamp": str(timestamp),
                                    "magnitude": str(magnitude),
                                    "percentage": str(percentage),
                                    "duration": str(duration),
                                    "columns": columns
                                }
                else: 
                    inj_params = None
                if selected_mode == "batch":
                    response = handler.handle_run_batch(selected_dataset, selected_detection_model, job_name, inj_params)
                else:
                    response = handler.handle_run_stream(selected_dataset, selected_detection_model, job_name, speedup, inj_params)
                style.update({"backgroundColor": "#4CAF50"})
            else:
                style.update({"backgroundColor": "#e74c3c"})
                children = "Job name already exists!"
            style.update({"display": "block"})
            return style, False, children
        elif trigger == "popup-interval.n_intervals":
            style.update({"display": "none"})
            return style, True, children 

        return style, True, children

def create_active_jobs(active_jobs):
    if len(active_jobs) == 0:
        return ["No active jobs found."]
    job_divs = []
    for job in active_jobs:
        new_div = html.Div([
            dcc.ConfirmDialog(
                id={"type": "confirm-box", "index": job["name"]},
                message=f'Are you sure you want to cancel the job {job["name"]}?',
                displayed=False,
            ),
            dcc.Link(
                children=[job["name"]],
                href=f'/{job["name"]}' if job["type"] == "stream" else f'/{job["name"]}?batch=True',
                style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
            ),
            html.Button("Stop", id={"type": "remove-dataset-btn", "index": job["name"]}, n_clicks=0, style={
                "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
            })
        ])
        job_divs.append(new_div)

    return [job_divs]