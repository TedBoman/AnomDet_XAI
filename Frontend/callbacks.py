from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update
import json
from get_handler import get_handler
#from bokeh.plotting import figure
#from bokeh.embed import file_html
#from bokeh.resources import CDN
import os

def get_index_callbacks(app):
    # --- Callback to toggle visibility of Injection panel ---
    @app.callback(
        Output("injection-panel", "style"), # Target the inner panel
        Input("injection-check", "value")
    )
    def toggle_injection_panel(selected_injection):
        if "use_injection" in selected_injection:
            # Return style to make it visible, keep other styles if needed
            return {"display": "block", "marginTop": "15px", "padding": "10px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88"}
        return {"display": "none"}

    # --- Callback to toggle visibility of Labeled panel ---
    @app.callback(
        Output("label-column-selection-div", "style"),
        Input("labeled-check", "value")
    )
    def toggle_labeled_panel(selected_labeled):
        if "is_labeled" in selected_labeled:
            return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}

    # --- Callback to toggle visibility of XAI options panel ---
    @app.callback(
        Output("xai-options-div", "style"),
        Input("xai-check", "value")
    )
    def toggle_xai_panel(selected_xai):
        if "use_xai" in selected_xai:
             return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}
    
    # --- Callback to populate columns for Injection Dropdown AND Label Dropdown ---
    @app.callback(
        Output("injection_column-dropdown", "options"),
        Output("label-column-dropdown", "options"),
        Output("label-column-dropdown", "value"),
        Input("dataset-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_column_dropdown(selected_dataset):
        print(f"--- update_column_dropdown callback triggered ---") # Start marker
        print(f"Selected Dataset: {selected_dataset!r}") # Use !r to clearly see None vs ""

        # Check if a dataset is actually selected
        if not selected_dataset:
            print("No dataset selected. Returning empty options.")
            return [], [], None

        handler = get_handler()
        print(f"Handler object: {handler}")
        columns = [] # Initialize columns
        options = [] # Initialize options
        try:
            print(f"Calling handler.handle_get_dataset_columns for '{selected_dataset}'...")
            columns = handler.handle_get_dataset_columns(selected_dataset)
            # *** Check what the handler actually returned ***
            print(f"Handler returned columns: {columns} (Type: {type(columns)})")

            # Ensure columns is a list before proceeding
            if not isinstance(columns, list):
                print("Warning: Handler did not return a list for columns. Returning empty options.")
                return [], [], None

            # Create options list
            options = [{"label": col, "value": col} for col in columns]
            print(f"Generated options for dropdowns: {options}")

            # Check if options are empty after filtering
            if not options:
                print("Warning: No column options remaining after filtering.")

            # Return options for both dropdowns, reset value for label dropdown
            return options, options, None

        except Exception as e:
            print(f"!!! ERROR inside update_column_dropdown try block: {e}")
            import traceback
            traceback.print_exc() # Print the full error traceback to the server console
            return [], [], None # Return empty on error

    # --- Callback to generate dynamic XAI settings panel ---
    @app.callback(
        Output("xai-settings-panel", "children"),
        Input("xai-method-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_xai_settings_panel(selected_xai_method):
        if not selected_xai_method or selected_xai_method == "none":
            return [] # Return empty if no method selected

        settings_children = [html.H5(f"Settings for {selected_xai_method.upper()}:", style={'color':'#ffffff', 'marginTop':'15px'})]

        # --- Use Pattern-Matching IDs ---
        if selected_xai_method == "shap":
            settings_children.extend([
                html.Div([
                    html.Label("Num Samples (nsamples):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                    dcc.Input(
                        # PATTERN MATCHING ID: type, method, param
                        id={'type': 'xai-setting', 'method': 'shap', 'param': 'nsamples'},
                        type="number", value=100, min=10, step=10, style={'width':'80px'}
                    )
                ], style={'marginBottom':'8px'}),
                html.Div([
                    html.Label("K for Background Summary (k_summary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                    dcc.Input(
                        id={'type': 'xai-setting', 'method': 'shap', 'param': 'k_summary'}, # Pattern ID
                        type="number", value=50, min=1, step=5, style={'width':'80px'}
                    )
                ], style={'marginBottom':'8px'}),
                html.Div([
                    html.Label("K for L1 Reg Features (l1_reg_k):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                    dcc.Input(
                        id={'type': 'xai-setting', 'method': 'shap', 'param': 'l1_reg_k'}, # Pattern ID
                        type="number", value=20, min=1, step=1, style={'width':'80px'}
                    )
                ], style={'marginBottom':'8px'})
            ])
        elif selected_xai_method == "lime":
            settings_children.extend([
                html.Div([
                    html.Label("Num Features to Explain:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                    dcc.Input(
                        id={'type': 'xai-setting', 'method': 'lime', 'param': 'num_features'}, # Pattern ID
                        type="number", value=15, min=1, step=1, style={'width':'80px'}
                    )
                ], style={'marginBottom':'8px'}),
                html.Div([
                    html.Label("Num Samples (Perturbations):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                    dcc.Input(
                        id={'type': 'xai-setting', 'method': 'lime', 'param': 'num_samples'}, # Pattern ID
                        type="number", value=1000, min=100, step=100, style={'width':'80px'}
                    )
                ], style={'marginBottom':'8px'})
            ])
        # Add elif for other methods

        return settings_children

    # --- Callback to toggle Speedup input based on mode ---
    @app.callback(
        Output("speedup-input-div", "style"),
        Input("mode-selection", "value")
    )
    def toggle_speedup_input(selected_mode):
        if selected_mode == "stream":
             return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}

    # --- Existing Callbacks for Active Jobs and Confirmation ---
    # Keep create_active_jobs function
    # Keep display_confirm callback
    # Keep manage_and_remove_active_jobs callback
    # Keep toggle_active_jobs_section callback
    # (Make sure their IDs still match the layout if anything changed)

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

    # --- start_job_handler ---
    @app.callback(
        [Output("popup", "style"), Output("popup-interval", "disabled"), Output("popup", "children")],
        [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
        [
            # Keep Existing States
            State("dataset-dropdown", "value"), State("detection-model-dropdown", "value"),
            State("mode-selection", "value"), State("name-input", "value"),
            State("injection-method-dropdown", "value"), State("timestamp-input", "value"),
            State("magnitude-input", "value"), State("percentage-input", "value"),
            State("duration-input", "value"), State("injection-column-dropdown", "value"),
            State("injection-check", "value"), State("speedup-input", "value"),
            State("popup", "style"),
            # Labeled States
            State("labeled-check", "value"), State("label-column-dropdown", "value"),
            # XAI Checkbox/Dropdown States
            State("xai-check", "value"), State("xai-method-dropdown", "value"),

            # --- NEW: Pattern-Matching State for ALL XAI settings ---
            # Get the 'value' property of all components matching the pattern
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'value'),
            # Get the corresponding 'id' dictionary for each value
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'id'),
        ]
    )
    def start_job_handler(
            n_clicks, n_intervals,
            selected_dataset, selected_detection_model, selected_mode, job_name,
            selected_injection_method, timestamp, magnitude, percentage, duration,
            injection_columns, inj_check, speedup, style,
            labeled_check_val, selected_label_col,
            xai_check_val, selected_xai_method,
            # --- NEW ARGS for pattern-matching states ---
            xai_settings_values, # List of values from matching components
            xai_settings_ids     # List of ID dictionaries from matching components
            ):
        handler = get_handler()
        children = "Job submission processed."
        style_copy = style.copy()

        ctx = callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'start-job-btn.n_clicks':
            if ctx.triggered and ctx.triggered[0]['prop_id'] == 'popup-interval.n_intervals':
                style_copy.update({"display": "none"})
                return style_copy, True, children
            return no_update, no_update, no_update

        trigger = ctx.triggered[0]["prop_id"]

        if trigger == "start-job-btn.n_clicks":
            # --- Basic Validation ---
            if not selected_dataset:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select a dataset."
            if not selected_detection_model:
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Please select a detection model."
            if not job_name:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Job name cannot be empty."

            response = handler.check_name(job_name)
            if response != "success":
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Job name already exists!"

            # --- Process Labeled Data Info ---
            is_labeled = "is_labeled" in labeled_check_val
            label_col_to_pass = None
            if is_labeled:
                if not selected_label_col:
                    style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                    return style_copy, False, "Please select the label column for the labeled dataset."
                label_col_to_pass = selected_label_col
            # --- End Labeled Data Info ---

            # --- Process Injection Info ---
            inj_params_list = None # Backend expects list or None
            if "use_injection" in inj_check:
                # Add validation for injection params if needed
                if not selected_injection_method or selected_injection_method == "None":
                     style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                     return style_copy, False, "Please select an injection method."
                if not timestamp:
                     style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                     return style_copy, False, "Please enter an injection timestamp."
                # Basic validation passed, create params dict
                inj_params = {
                    "anomaly_type": selected_injection_method, "timestamp": str(timestamp),
                    "magnitude": str(magnitude if magnitude is not None else 1), # Use default if None
                    "percentage": str(percentage if percentage is not None else 0), # Default if None
                    "duration": str(duration if duration else '0s'), # Default if None/empty
                    "columns": injection_columns if injection_columns else [] # Use empty list if None
                }
                inj_params_list = [inj_params] # Backend expects a list
            # --- End Injection Info ---

            # --- Process XAI Info ---
        use_xai = "use_xai" in xai_check_val
        xai_params = None
        if use_xai:
            if not selected_xai_method or selected_xai_method == "none":
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select an XAI method if 'Use Explainability' is checked."

            # --- Parse Pattern-Matching State results ---
            # Only parse settings for the currently selected method that are rendered
            xai_settings = {}
            print(f"DEBUG: Received XAI settings IDs: {xai_settings_ids}")
            print(f"DEBUG: Received XAI settings Values: {xai_settings_values}")

            for id_dict, value in zip(xai_settings_ids, xai_settings_values):
                # Check if the setting belongs to the currently selected method
                if id_dict['method'] == selected_xai_method:
                    param_name = id_dict['param']
                    # Provide defaults if input value is None (e.g., user cleared the input)
                    if value is None:
                        print(f"Warning: XAI setting '{param_name}' for method '{selected_xai_method}' has None value. Using default.")
                        # Assign defaults based on param name - keep these consistent!
                        if param_name == 'nsamples': value = 100
                        elif param_name == 'k_summary': value = 50
                        elif param_name == 'l1_reg_k': value = 20 # Match ID used
                        elif param_name == 'num_features': value = 15
                        elif param_name == 'num_samples': value = 1000
                        else: value = None # Or raise error for unknown param without default

                    xai_settings[param_name] = value

            print(f"DEBUG: Parsed XAI settings for '{selected_xai_method}': {xai_settings}")

            # --- Construct final xai_params ---
            # NOTE: Rename l1_reg_k to l1_reg_k_features if backend expects that specific name
            if selected_xai_method == "shap" and "l1_reg_k" in xai_settings:
                xai_settings["l1_reg_k_features"] = xai_settings.pop("l1_reg_k") # Rename key for backend

            xai_params = {
                "method": selected_xai_method,
                "settings": xai_settings # Contains parsed settings for the selected method
            }
            print(f"DEBUG: Constructed xai_params: {xai_params}")
        # --- End XAI Info ---

        # --- Call Backend Handler ---
        try:
            # Pass label_col_to_pass and xai_params to your backend handler
            if selected_mode == "batch":
                response = handler.handle_run_batch(
                    selected_dataset, selected_detection_model, job_name,
                    label_column=label_col_to_pass, xai_params=xai_params, inj_params=inj_params_list)
            else: # stream
                response = handler.handle_run_stream(
                    selected_dataset, selected_detection_model, job_name, speedup,
                    label_column=label_col_to_pass, xai_params=xai_params, inj_params=inj_params_list)

            if response == "success":
                style_copy.update({"backgroundColor": "#4CAF50", "display": "block"})
                children = f"Job '{job_name}' started successfully!"
            else:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                children = f"Backend error starting job: {response}"

        except Exception as e:
            print(f"Error calling backend handler: {e}")
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            children = "Error communicating with backend."

        return style_copy, False, children # Show popup

def create_active_jobs(active_jobs):
    if len(active_jobs) == 0:
        return ["No active jobs found."]
    job_divs = []
    GRAFANA_URL = f"http://localhost:{os.getenv('GRAFANA_PORT')}"
    for job in active_jobs:
        new_div = html.Div([
            dcc.ConfirmDialog(
                id={"type": "confirm-box", "index": job["name"]},
                message=f'Are you sure you want to cancel the job {job["name"]}?',
                displayed=False,
            ),
            html.A(
                children=[job["name"]],
                # href=f'/{job["name"]}' if job["type"] == "stream" else f'/{job["name"]}?batch=True', #Old version with bokeh
                href=f'{GRAFANA_URL}/d/stream01/stream-jobs' if job["type"] == "stream" else f'{GRAFANA_URL}/d/batch01/batch-jobs', # New version with grafana
                style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
            ),
            html.Button("Stop", id={"type": "remove-dataset-btn", "index": job["name"]}, n_clicks=0, style={
                "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
            })
        ])
        job_divs.append(new_div)

    return [job_divs]