# callbacks.py (Complete and Updated File)

import sys
import traceback
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update
import json
from get_handler import get_handler
import os 

# --- Create_active_jobs FUNCTION ---
def create_active_jobs(active_jobs):
    """
    Generates the HTML structure for the active jobs list, returning a single Div.
    """
    if not active_jobs:
        # Return a single Div containing the message
        return html.Div("No active jobs found.")

    job_divs = []
    for job in active_jobs:
        job_name = job.get("name", "Unknown Job") # Use .get for safety
        # Construct the internal Dash link
        dash_link = f'/job/{job_name}'

        # Create the Div for each job entry
        job_entry = html.Div([
            # Confirmation dialog for stopping the job
            dcc.ConfirmDialog(
                id={"type": "confirm-box", "index": job_name},
                message=f'Are you sure you want to cancel the job: {job_name}?',
                displayed=False,
            ),
            # Link to the job's results page within the Dash app
            html.A(
                children=[job_name],
                href=dash_link,
                # target="_blank", # Optional: uncomment to open in new tab
                style={
                    "marginRight": "15px",
                    "color": "#4CAF50", # Green link color
                    "textDecoration": "none",
                    "fontWeight": "bold",
                    "fontSize": "16px"
                }
            ),
            # Button to stop/cancel the job
            html.Button(
                "Stop Job",
                id={"type": "remove-dataset-btn", "index": job_name},
                n_clicks=0,
                style={
                    "fontSize": "12px",
                    "backgroundColor": "#e74c3c", # Red button color
                    "color": "#ffffff", # White text
                    "border": "none",
                    "borderRadius": "5px",
                    "padding": "5px 10px", # Slightly more padding
                    "cursor": "pointer" # Indicate it's clickable
                }
            )
        ], style={'paddingBottom': '8px', 'borderBottom': '1px solid #444'}) # Add padding and separator line

        job_divs.append(job_entry)

    # Return a single Div wrapping the title and the list container
    return html.Div([
        html.H4("Active Job List", style={'color': '#C0C0C0', 'marginBottom': '10px'}),
        html.Div(job_divs) # The list of job divs is the second child
    ])

# --- Callbacks for index page ---
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
        Output("label-column-dropdown", "value"), # Reset label value when dataset changes
        Input("dataset-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_column_dropdown(selected_dataset):
        print(f"--- update_column_dropdown callback triggered ---")
        print(f"Selected Dataset: {selected_dataset!r}")

        if not selected_dataset:
            print("No dataset selected. Returning empty options.")
            return [], [], None

        handler = get_handler()
        print(f"Handler object: {handler}")
        columns = []
        options = []
        try:
            print(f"Calling handler.handle_get_dataset_columns for '{selected_dataset}'...")
            columns = handler.handle_get_dataset_columns(selected_dataset)
            print(f"Handler returned columns: {columns} (Type: {type(columns)})")

            if not isinstance(columns, list):
                print("Warning: Handler did not return a list for columns. Returning empty options.")
                return [], [], None

            options = [{"label": col, "value": col} for col in columns]
            print(f"Generated options for dropdowns: {options}")

            if not options: print("Warning: No column options remaining.")

            return options, options, None # Return options for both, reset label value

        except Exception as e:
            print(f"!!! ERROR inside update_column_dropdown try block: {e}")
            traceback.print_exc()
            return [], [], None

    # --- Callback to generate dynamic XAI settings panel ---
    @app.callback(
        Output("xai-settings-panel", "children"),
        Input("xai-method-dropdown", "value"),
        State("dataset-dropdown", "value"), # Need dataset to potentially populate features
    )
    def update_xai_settings_panel(selected_xai_methods, selected_dataset):
        if not selected_xai_methods:
            return []

        active_methods = [m for m in selected_xai_methods if m != 'none']
        if not active_methods:
             return []

        all_settings_children = []
        handler = get_handler() # Get handler once if needed for multiple methods

        # Fetch columns once if needed by multiple XAI methods (like DiCE)
        column_options = []
        if selected_dataset:
            try:
                columns = handler.handle_get_dataset_columns(selected_dataset)
                if isinstance(columns, list):
                    column_options = [{"label": col, "value": col} for col in columns]
                else:
                    print(f"Warning: Handler did not return list for columns: {columns}")
            except Exception as e:
                print(f"!!! ERROR fetching columns for XAI settings panel: {e}")
                traceback.print_exc()

        # --- Loop through each selected method ---
        for i, selected_xai_method in enumerate(active_methods):
            if i > 0: all_settings_children.append(html.Hr(style={'borderColor': '#555', 'margin': '20px 0'}))

            method_settings = [html.H5(f"Settings for {selected_xai_method.upper()}:", style={'color':'#ffffff', 'marginTop':'15px', 'marginBottom': '10px'})]

            # --- Pattern-Matching IDs ---
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
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Explainer method:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'shap_method'}, options=[{'label': 'KernelShap (default)', 'value': 'kernel'},{'label': 'TreeShap', 'value': 'tree'},{'label': 'LinearShap', 'value': 'linear'},{'label': 'PartitionShap', 'value': 'partition'}], value='kernel', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
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
                    # Features to vary dropdown
                    html.Div([
                        html.Label("Features to vary (features_to_vary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(
                            id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'features_to_vary'},
                            options=column_options, # Use pre-fetched options
                            value=[], # Default to empty list (vary all mutable by default in DiCE)
                            multi=True,
                            placeholder="Select features (leave empty to vary all mutable)",
                            style={'width': '90%', 'maxWidth':'500px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}
                        )
                    ], style={'marginBottom':'8px'}),
                    # Other DiCE settings
                     html.Div([
                          html.Label("Backend (ML model framework):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'backend'}, options=[{'label': 'SciKit-Learn', 'value': 'sklearn'},{'label': 'Tensorflow 1', 'value': 'TF1'},{'label': 'Tensorflow 2', 'value': 'TF2'},{'label': 'PyTorch', 'value': 'pytorch'}], value='sklearn', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                          html.Label("DiCE Method (dice_method):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'dice_method'}, options=[{'label': 'Random', 'value': 'random'},{'label': 'Genetic', 'value': 'genetic'},{'label': 'KD-Tree', 'value': 'kdtree'}], value='genetic', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'})
                ]
                method_settings.extend(dice_specific_settings)
            # Add elif for other methods...

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

    # --- Callback to toggle visibility of Active Jobs section ---
    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-jobs-list", "children") # Trigger based on content changes
    )
    def toggle_active_jobs_section(children):
        # Show the active jobs section unless the content indicates no jobs
        # Need to be careful about comparing component structures vs simple strings
        no_jobs_message = "No active jobs found."
        display_style = {"display": "block", "marginTop": "30px"}
        hide_style = {"display": "none"}

        if isinstance(children, list) and len(children) > 0:
            # Check if the first child is a Div containing the no_jobs_message
            # This depends on how create_active_jobs returns the message now
            first_child = children[0]
            if isinstance(first_child, html.Div) and getattr(first_child, 'children', None) == no_jobs_message:
                 return hide_style
            else:
                 return display_style # Assume list contains job divs
        elif isinstance(children, str) and children == no_jobs_message:
            return hide_style # Direct string comparison
        elif children: # If children exist and are not the 'no jobs' message
             return display_style
        else: # If children are None or empty list
             return hide_style


    # --- Callback to display confirmation box for stopping a job ---
    @app.callback(
        Output({"type": "confirm-box", "index": MATCH}, "displayed"),
        Input({"type": "remove-dataset-btn", "index": MATCH}, "n_clicks"),
        prevent_initial_call=True # Don't display on page load
    )
    def display_confirm(n_clicks):
        return True if n_clicks and n_clicks > 0 else False

    # --- Callback to manage active jobs list ---
    @callback(
        Output("active-jobs-list", "children"),
        # Output("active-jobs-error-message", "children"), # Keep commented unless added
        Input("job-interval", "n_intervals"),
        Input({"type": "confirm-box", "index": ALL}, "submit_n_clicks"), # Listen to confirmation clicks
        State("active-jobs-json", "data") # Store previous state as JSON
    )
    def manage_and_remove_active_jobs(n_intervals, submit_n_clicks_list, active_jobs_json_state):
        """
        Periodically fetches the list of active jobs and updates the display.
        Also handles job cancellation confirmation. Includes error handling.
        Returns a list containing one item for the single Output.
        """
        ctx = callback_context
        triggered_id = ctx.triggered_id
        print(f"manage_and_remove_active_jobs triggered by: {triggered_id}") # Log trigger

        handler = get_handler()
        error_message = None # Initialize error message

        # --- Handle Job Cancellation ---
        # Check if the trigger was one of the confirmation buttons AND it was clicked
        if isinstance(triggered_id, dict) and triggered_id.get("type") == "confirm-box":
            # Find which button was clicked
            button_index = -1
            for i, n_clicks in enumerate(submit_n_clicks_list):
                 # Check if this specific confirmation box was clicked (n_clicks > 0)
                 # This logic assumes submit_n_clicks resets; adjust if needed
                 if n_clicks and n_clicks > 0:
                     # Extract the job name from the ID of the confirmation box that triggered
                     all_confirm_ids = ctx.inputs_list[1] # Get list of Input dicts for confirm-box
                     if i < len(all_confirm_ids):
                          button_index = i
                          job_to_cancel = all_confirm_ids[i]['id']['index']
                          print(f"Confirmation received for job: {job_to_cancel}")
                          try:
                              response = handler.handle_cancel_job(job_to_cancel)
                              if response != "success":
                                   print(f"Backend error cancelling job '{job_to_cancel}': {response}")
                                   error_message = f"Error cancelling {job_to_cancel}: {response}"
                              # Reset n_clicks? Dash doesn't directly support this easily
                              # The callback will proceed to refresh the list anyway
                          except Exception as cancel_err:
                              print(f"!!! EXCEPTION during handle_cancel_job for '{job_to_cancel}': {cancel_err}")
                              traceback.print_exc()
                              error_message = f"Frontend error cancelling job {job_to_cancel}."
                          break # Assume only one confirmation can be submitted at a time

        # --- Fetch and Update Active Jobs List ---
        try:
            print("Fetching active jobs from backend...")
            raw_response = handler.handle_get_running()
            print(f"Raw response from handle_get_running: {raw_response}") # Log raw response

            if not raw_response: raise ValueError("Received empty response from handle_get_running.")

            active_jobs_data = json.loads(raw_response)
            print(f"Parsed active_jobs_data: {active_jobs_data}") # Log parsed data

            if not isinstance(active_jobs_data, dict) or 'running' not in active_jobs_data: raise TypeError("Invalid data structure received. Expected {'running': [...]}")
            if not isinstance(active_jobs_data['running'], list): raise TypeError("Invalid data structure: 'running' key is not a list.")
            active_jobs_list = active_jobs_data["running"]
            print(f"Extracted active_jobs_list: {active_jobs_list}") # Log the final list

            # --- Compare with previous state ---
            current_jobs_json = json.dumps(active_jobs_list, sort_keys=True) # Sort keys for consistent comparison
            prev_jobs_json_state = active_jobs_json_state if active_jobs_json_state else json.dumps([]) # Handle initial None state
            # Re-parse prev_jobs_json_state for comparison consistency if needed, or compare strings directly
            # For simplicity, compare JSON strings directly
            if current_jobs_json == prev_jobs_json_state and triggered_id == "job-interval":
                print("Job list hasn't changed. Returning no_update.")
                return no_update

            # --- Generate new layout component ---
            print("Job list changed or cancellation may have occurred. Updating display.")
            new_children_component = create_active_jobs(active_jobs_list) # Gets the single Div

            # --- Wrap the single component in a list for the Output ---
            return [new_children_component]

        except Exception as e:
            print(f"!!! EXCEPTION in manage_and_remove_active_jobs callback: {e}")
            traceback.print_exc()
            error_output = html.Div([
                html.P("Error updating active jobs list:", style={'color': 'red', 'fontWeight': 'bold'}),
                html.Pre(f"{traceback.format_exc()}", style={'color': 'red', 'fontSize': 'small', 'whiteSpace': 'pre-wrap'})
            ])
            # --- Wrap the error component in a list for the Output ---
            return [error_output]


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
            State("labeled-check", "value"), State("label-column-dropdown", "value"),
            State("xai-check", "value"), State("xai-method-dropdown", "value"),
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
            xai_check_val, selected_xai_methods,
            xai_settings_values, xai_settings_ids
            ):
        handler = get_handler()
        children = "Job submission processed."
        style_copy = style.copy() if style else {} # Ensure style_copy is a dict

        ctx = callback_context
        # Check if callback was triggered by button click or interval timeout
        triggered_prop_id = ctx.triggered[0]['prop_id'] if ctx.triggered else 'No trigger'

        # Handle popup closing
        if triggered_prop_id == 'popup-interval.n_intervals':
            style_copy.update({"display": "none"})
            # Return style, disable interval, keep children text
            return style_copy, True, children

        # Handle button click
        if triggered_prop_id != 'start-job-btn.n_clicks' or not n_clicks or n_clicks == 0:
            # If not triggered by button or button hasn't been clicked, do nothing
            return no_update, no_update, no_update

        # --- Proceed with Job Submission Logic (triggered by button) ---
        print(f"Start job button clicked (n_clicks={n_clicks})")

        # Basic Validation
        error_msg = None
        if not selected_dataset: error_msg = "Please select a dataset."
        elif not selected_detection_model: error_msg = "Please select a detection model."
        elif not job_name: error_msg = "Job name cannot be empty."
        else:
            response = handler.check_name(job_name)
            if response != "success": error_msg = "Job name already exists!"

        if error_msg:
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            return style_copy, False, error_msg # Show error popup

        # Process Labeled Data Info
        is_labeled = "is_labeled" in labeled_check_val
        label_col_to_pass = None
        if is_labeled:
            if not selected_label_col:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select the label column."
            label_col_to_pass = selected_label_col

        # Process Injection Info
        inj_params_list = None
        if "use_injection" in inj_check:
            if not selected_injection_method or selected_injection_method == "None": error_msg = "Please select an injection method."
            elif not timestamp: error_msg = "Please enter an injection timestamp."
            # Add more validation for magnitude, percentage, duration, columns if needed

            if error_msg:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, error_msg

            inj_params = {
                "anomaly_type": selected_injection_method, "timestamp": str(timestamp),
                "magnitude": str(magnitude if magnitude is not None else 1),
                "percentage": str(percentage if percentage is not None else 0),
                "duration": str(duration if duration else '0s'),
                "columns": injection_columns if injection_columns else []
            }
            inj_params_list = [inj_params]

        # Process XAI Info
        use_xai = "use_xai" in xai_check_val
        xai_params_list = None
        if use_xai:
            active_methods = [m for m in selected_xai_methods if m != 'none'] if selected_xai_methods else []
            if not active_methods:
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Please select at least one XAI method."

            all_parsed_settings = {}
            print(f"DEBUG: Received XAI settings IDs: {xai_settings_ids}")
            print(f"DEBUG: Received XAI settings Values: {xai_settings_values}")

            for id_dict, value in zip(xai_settings_ids, xai_settings_values):
                method_name = id_dict.get('method')
                param_name = id_dict.get('param')
                if not method_name or not param_name or method_name not in active_methods: continue

                if method_name not in all_parsed_settings: all_parsed_settings[method_name] = {}
                # Handle None values if necessary (user cleared input), similar to original logic
                # ... (add specific default logic if needed) ...
                all_parsed_settings[method_name][param_name] = value

            print(f"DEBUG: Parsed all XAI settings: {all_parsed_settings}")

            xai_params_list = []
            for method_name in active_methods:
                if method_name in all_parsed_settings:
                    current_settings = all_parsed_settings[method_name]
                    # Perform key renaming if needed (e.g., l1_reg_k for Shap)
                    if method_name == "ShapExplainer" and "l1_reg_k" in current_settings:
                        current_settings["l1_reg_k_features"] = current_settings.pop("l1_reg_k")

                    xai_params_list.append({"method": method_name, "settings": current_settings})
                else:
                    print(f"Warning: No settings found/parsed for selected method: {method_name}")
                    xai_params_list.append({"method": method_name, "settings": {}}) # Add with empty settings

            if not xai_params_list: # Should not happen if active_methods is not empty
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Error constructing XAI parameters."

            print(f"DEBUG: Constructed xai_params_list for backend: {xai_params_list}")


        # Call Backend Handler
        try:
            print(f"Sending job '{job_name}' with mode '{selected_mode}'...")
            print(f"  Dataset: {selected_dataset}, Model: {selected_detection_model}")
            print(f"  Label Column: {label_col_to_pass}")
            print(f"  XAI Params: {xai_params_list}")
            print(f"  Injection Params: {inj_params_list}")
            sys.stdout.flush()

            if selected_mode == "batch":
                response = handler.handle_run_batch(
                    selected_dataset, selected_detection_model, job_name,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list)
            else: # stream
                # Validate speedup for stream mode
                speedup_val = 1.0 # Default
                try:
                     speedup_val = float(speedup) if speedup is not None else 1.0
                     if speedup_val <= 0: raise ValueError("Speedup must be positive.")
                except ValueError:
                     style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                     return style_copy, False, "Invalid speedup value for stream mode."

                response = handler.handle_run_stream(
                    selected_dataset, selected_detection_model, job_name, speedup_val,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list)

            if response == "success":
                style_copy.update({"backgroundColor": "#4CAF50", "display": "block"})
                children = f"Job '{job_name}' started successfully!"
            else:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                children = f"Backend error starting job: {response}"

        except Exception as e:
            print(f"Error calling backend handler: {e}")
            traceback.print_exc()
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            children = "Error communicating with backend."

        # Return style to show popup, disable interval timer, set popup text
        return style_copy, False, children

