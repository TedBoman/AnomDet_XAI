import traceback
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update
import json
from get_handler import get_handler
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
        Output("injection-column-dropdown", "options"),
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
        State("dataset-dropdown", "value"),
    )
    def update_xai_settings_panel(selected_xai_methods, selected_dataset):
        if not selected_xai_methods:
            return [] # Return empty if no method selected
        
        active_methods = [m for m in selected_xai_methods if m != 'none']
        if not active_methods:
             return []

        all_settings_children = [] # Initialize list to hold all settings components
        
        # --- Loop through each selected method ---
        for i, selected_xai_method in enumerate(active_methods):
            # Add a separator between methods if more than one is selected
            if i > 0:
                all_settings_children.append(html.Hr(style={'borderColor': '#555', 'margin': '20px 0'}))

            # Add heading for the current method
            method_settings = [html.H5(f"Settings for {selected_xai_method.upper()}:", style={'color':'#ffffff', 'marginTop':'15px', 'marginBottom': '10px'})]

            # --- Use Pattern-Matching IDs ---
            if selected_xai_method == "ShapExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'n_explain_max'}, type="number", value=100, min=10, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (nsamples):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'nsamples'}, type="number", value=100, min=10, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for Background Summary (k_summary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'k_summary'}, type="number", value=50, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for L1 Reg Features (l1_reg_k):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'l1_reg_k'}, type="number", value=20, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "LimeExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Features to Explain:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_features'}, type="number", value=15, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (Perturbations):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_samples'}, type="number", value=1000, min=100, step=100, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Kernel Width (kernel_width):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'kernel_width'}, type="number", placeholder="LIME default", min=0.01, step=0.1, style={'width':'110px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Feature Selection:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'feature_selection'}, options=[{'label': 'Auto', 'value': 'auto'},{'label': 'Highest Weights', 'value': 'highest_weights'},{'label': 'Forward Selection', 'value': 'forward_selection'},{'label': 'Lasso Path', 'value': 'lasso_path'},{'label': 'None', 'value': 'none'}], value='auto', clearable=False, style={'width': '180px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Discretize Continuous:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'discretize_continuous'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Sample Around Instance:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'sample_around_instance'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "DiceExplainer":
                dice_specific_settings = [
                    html.Div([
                         html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                         html.Label("Num Counterfactuals (total_CFs):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'total_CFs'}, type="number", value=5, min=1, step=1, style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                         html.Label("Desired Class (desired_class):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'desired_class'}, type="text", value="opposite", style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),

                    # --- Dynamic Features to Vary Dropdown ---
                    html.Div([
                        html.Label("Features to vary (features_to_vary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(
                            # Use a specific index in MATCH perhaps? Or just ensure unique IDs if needed elsewhere
                            # For pattern matching state collection, the dict id is sufficient
                            id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'features_to_vary'},
                            options=[], # To be populated below
                            value=[],
                            multi=True,
                            placeholder="Select features (leave empty to vary all)",
                            style={'width': '90%', 'maxWidth':'500px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}
                        )
                    ], id=f'dice-features-vary-div-{selected_xai_method}', # Make ID unique if needed elsewhere
                       style={'marginBottom':'8px'}), # Unique ID might not be needed if only accessed via pattern matching

                    html.Div([
                          html.Label("Backend (ML model framework):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'backend'}, options=[{'label': 'SciKit-Learn', 'value': 'sklearn'},{'label': 'Tensorflow 1', 'value': 'TF1'},{'label': 'Tensorflow 2', 'value': 'TF2'},{'label': 'PyTorch', 'value': 'pytorch'}], value='sklearn', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                          html.Label("DiCE Method (dice_method):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'dice_method'}, options=[{'label': 'Random', 'value': 'random'},{'label': 'Genetic', 'value': 'genetic'},{'label': 'KD-Tree', 'value': 'kdtree'}], value='genetic', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'})
                ]

                # --- Populate the features_to_vary dropdown options ---
                column_options = []
                if selected_dataset:
                    handler = get_handler()
                    try:
                        columns = handler.handle_get_dataset_columns(selected_dataset)
                        if isinstance(columns, list):
                            column_options = [{"label": col, "value": col} for col in columns]
                        else:
                            print(f"Warning: Handler did not return list for columns: {columns}")
                    except Exception as e:
                        print(f"!!! ERROR fetching columns for DiceExplainer: {e}")
                        traceback.print_exc()

                # Find the dropdown within dice_specific_settings and assign options
                # This assumes the dropdown is the second child of the Div with label "Features to vary..."
                for component in dice_specific_settings:
                     if isinstance(component, html.Div) and component.children and isinstance(component.children[0], html.Label):
                          if component.children[0].children == "Features to vary (features_to_vary):":
                               if len(component.children) > 1 and isinstance(component.children[1], dcc.Dropdown):
                                    component.children[1].options = column_options
                                    break

                method_settings.extend(dice_specific_settings)
            # Add elif for other methods
            # Append the generated settings for this method to the main list
            all_settings_children.extend(method_settings)

        return all_settings_children

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
            State("dataset-dropdown", "value"), State("detection-model-dropdown", "value"),
            State("mode-selection", "value"), State("name-input", "value"),
            State("injection-method-dropdown", "value"), State("timestamp-input", "value"),
            State("magnitude-input", "value"), State("percentage-input", "value"),
            State("duration-input", "value"), State("injection-column-dropdown", "value"),
            State("injection-check", "value"), State("speedup-input", "value"),
            State("popup", "style"),
            # Labeled States
            State("labeled-check", "value"), State("label-column-dropdown", "value"),
            # --- XAI States (MODIFIED) ---
            State("xai-check", "value"),
            State("xai-method-dropdown", "value"), # Receives a LIST now
            # --- Pattern-Matching State for ALL XAI settings ---
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'value'),
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'id'),
        ]
    )
    def start_job_handler(
            n_clicks, n_intervals,
            selected_dataset, selected_detection_model, selected_mode, job_name,
            selected_injection_method, timestamp, magnitude, percentage, duration,
            injection_columns, inj_check, speedup, style,
            labeled_check_val, selected_label_col,
            # --- ARGS for pattern-matching states ---
            xai_check_val,
            selected_xai_methods,
            xai_settings_values,
            xai_settings_ids
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
        xai_params_list = None # Changed from xai_params to list

        if use_xai:
            active_methods = [m for m in selected_xai_methods if m != 'none'] if selected_xai_methods else []
            if not active_methods:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select at least one XAI method if 'Use Explainability' is checked."

            # --- Parse ALL Pattern-Matching State results into a structured dict ---
            # all_parsed_settings = { 'ShapExplainer': {'param1': val1, ...}, 'DiceExplainer': {'paramA': valA,...} }
            all_parsed_settings = {}
            print(f"DEBUG: Received XAI settings IDs: {xai_settings_ids}")
            print(f"DEBUG: Received XAI settings Values: {xai_settings_values}")

            for id_dict, value in zip(xai_settings_ids, xai_settings_values):
                method_name = id_dict['method']
                param_name = id_dict['param']

                # Only process settings for methods that are actually selected
                if method_name not in active_methods:
                    continue # Skip settings for methods not currently selected

                # Initialize dict for the method if it doesn't exist
                if method_name not in all_parsed_settings:
                    all_parsed_settings[method_name] = {}

                # Handle None values if necessary (user cleared input)
                # (Add specific default logic here if needed, similar to previous single-method version)
                if value is None and param_name not in ['features_to_vary', 'kernel_width']: # Allow None/empty for these
                     print(f"Warning: XAI setting '{param_name}' for method '{method_name}' has None value. Check defaults.")
                     # Example default assignment:
                     # if method_name == 'LimeExplainer' and param_name == 'num_samples': value = 1000

                # Special handling / Type conversion if needed
                if param_name == 'features_to_vary' and value == []:
                    print(f"DEBUG: features_to_vary for {method_name} is empty list. Backend might default to 'all'.")

                # Store the value
                all_parsed_settings[method_name][param_name] = value

            print(f"DEBUG: Parsed all XAI settings: {all_parsed_settings}")

            # --- Construct the final list of XAI params for the backend ---
            xai_params_list = []
            for method_name in active_methods:
                if method_name in all_parsed_settings:
                    current_settings = all_parsed_settings[method_name]

                    # Perform any key renaming needed for the backend *for this specific method*
                    if method_name == "ShapExplainer" and "l1_reg_k" in current_settings:
                        current_settings["l1_reg_k_features"] = current_settings.pop("l1_reg_k")

                    xai_params_list.append({
                        "method": method_name,
                        "settings": current_settings
                    })
                else:
                    # This case might happen if a method was selected but somehow its settings weren't rendered/parsed
                    print(f"Warning: No settings found/parsed for selected method: {method_name}")
                    # Decide how to handle: skip, add with empty settings, or error out?
                    # Example: Add with empty settings
                    # xai_params_list.append({"method": method_name, "settings": {}})


            if not xai_params_list: # If loop finished but list is empty (e.g., due to warnings/skips)
                 print("Error: Could not construct XAI parameters for selected methods.")
                 # Handle error appropriately

            print(f"DEBUG: Constructed xai_params_list for backend: {xai_params_list}")
        # --- End XAI Info ---

        # --- Call Backend Handler ---
        try:
            # Pass label_col_to_pass and xai_params to your backend handler
            if selected_mode == "batch":
                response = handler.handle_run_batch(
                    selected_dataset, selected_detection_model, job_name,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list)
            else: # stream
                response = handler.handle_run_stream(
                    selected_dataset, selected_detection_model, job_name, speedup,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list)

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