# job_page_callbacks.py
import json
import sys
import os
import urllib.parse

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import plotly.graph_objects as go
import pandas as pd
import traceback
from datetime import datetime, timedelta, timezone
from io import StringIO 
from pages.job_page import get_display_job_name
from get_handler import get_handler 

# --- Ensure XAI_DIR is consistent with app.py ---
XAI_DIR = "/app/data" # The path INSIDE the container
# ---------------------------------------------

# --- Helper Function to Generate Asset URL ---
def get_asset_url(job_name, method_name, filename):
    """Constructs the URL for accessing assets via the Flask static route."""
    # Quote each part to handle special characters in names
    quoted_job = urllib.parse.quote(job_name)
    quoted_method = urllib.parse.quote(method_name)
    quoted_file = urllib.parse.quote(filename)
    return f"/xai-assets/{quoted_job}/{quoted_method}/{quoted_file}"
# -------------------------------------------

# --- Helper function to format nested dicts/lists prettily ---
def create_pretty_dict_list_display(data, indent=0):
    """Recursively creates html.Divs/Lists to display nested data."""
    items = []
    indent_space = "  " * indent # Using non-breaking space for indent
    if isinstance(data, dict):
        for key, value in data.items():
            # Display key
            item_content = [html.Span(f"{indent_space}{key}: ", style={'fontWeight': 'bold'})]
            # Display value (recurse if nested)
            if isinstance(value, (dict, list)):
                 # Add newline before nested structure for clarity
                item_content.append(html.Br())
                item_content.append(create_pretty_dict_list_display(value, indent + 1))
            else:
                item_content.append(html.Span(f"{value}"))
            items.append(html.Div(item_content))
    elif isinstance(data, list):
         # Special handling for list of dicts (like xai_params)
        is_list_of_dicts = all(isinstance(i, dict) for i in data)
        for index, value in enumerate(data):
            if is_list_of_dicts:
                 # Add a separator/header for each item in the list
                 items.append(html.Div(f"{indent_space}Item {index+1}:", style={'marginTop': '5px', 'fontStyle':'italic'}))
                 items.append(create_pretty_dict_list_display(value, indent + 1))
            elif isinstance(value, (dict, list)):
                items.append(create_pretty_dict_list_display(value, indent + 1))
            else:
                items.append(html.Div(f"{indent_space}- {value}"))
    return html.Div(items)

# --- Helper function to create a simple key-value table section ---
def create_info_section(title, data_dict, theme_colors, format_floats=True):
    """Creates a styled Div with H4 title and key-value pairs."""
    rows = []
    for key, value in data_dict.items():
        # Format float values nicely
        if format_floats and isinstance(value, float):
            display_value = f"{value:.4f}" # Adjust precision as needed
        else:
            display_value = str(value)

        # Improve readability of keys
        display_key = key.replace('_', ' ').title()

        rows.append(html.Div([
            html.Span(f"{display_key}:", style={'fontWeight': 'bold', 'minWidth': '200px', 'display': 'inline-block'}),
            html.Span(display_value)
        ], style={'marginBottom': '5px'}))

    return html.Div([
        html.H4(title, style={'borderBottom': f"1px solid {theme_colors.get('border_light', '#555')}", 'paddingBottom': '5px', 'marginTop': '15px', 'marginBottom': '10px', 'color': 'white'}),
        *rows
    ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'}) # Slightly different background


# --- Helper Function for CSV Display ---
def create_datatable(file_path):
    """Reads a CSV and returns a Dash DataTable or an error message."""
    try:
        # Use StringIO to handle potential encoding issues if needed, though direct path usually works
        df = pd.read_csv(file_path)
        # Limit rows for display performance if necessary
        max_rows = 50
        if len(df) > max_rows:
             df_display = df.head(max_rows)
             disclaimer = html.P(f"(Displaying first {max_rows} rows)", style={'fontSize':'small', 'color':'#ccc'})
        else:
             df_display = df
             disclaimer = None

        table = dash_table.DataTable(
             columns=[{"name": i, "id": i} for i in df_display.columns],
             data=df_display.to_dict('records'),
             style_table={'overflowX': 'auto', 'marginTop': '10px'},
             style_header={
                 'backgroundColor': 'rgb(30, 30, 30)',
                 'color': 'white',
                 'fontWeight': 'bold'
             },
             style_cell={
                 'backgroundColor': 'rgb(50, 50, 50)',
                 'color': 'white',
                 'border': '1px solid #555',
                 'textAlign': 'left',
                 'padding': '5px',
                 'minWidth': '80px', 'width': '150px', 'maxWidth': '300px', # Adjust width constraints
                 'overflow': 'hidden',
                 'textOverflow': 'ellipsis',
             },
             tooltip_data=[ # Add tooltips for potentially truncated cells
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in df_display.to_dict('records')
            ],
            tooltip_duration=None # Keep tooltip visible indefinitely on hover
        )
        if disclaimer:
             return html.Div([disclaimer, table])
        else:
             return table
    except Exception as e:
        print(f"Error reading/parsing CSV {file_path}: {e}")
        return html.P(f"Error displaying CSV: {os.path.basename(file_path)} - {e}", style={'color':'red'})
# ------------------------------------

# --- Helper Function for counterfactual CSV Display ---
def create_cfe_delta_table(file_path):
    """
    Reads a CFE CSV and returns a single styled DataTable containing both
    the original row and counterfactual rows, highlighting differences.
    """
    try:
        df_cfe = pd.read_csv(file_path)

        # --- Find Original and Counterfactual Data ---
        original_rows = df_cfe[df_cfe['type'].str.lower() == 'original']
        if original_rows.empty:
            return html.P(f"Error: 'original' row not found in {os.path.basename(file_path)}.", style={'color': 'red'})
        original_row = original_rows.iloc[0] # Use the first original row found

        cf_rows = df_cfe[df_cfe['type'].str.lower() == 'counterfactual']
        # No error if cf_rows is empty, we'll just show the original

        # Identify feature columns (exclude 'type', but keep others like 'label')
        all_display_cols = list(df_cfe.columns)
        feature_cols = [col for col in all_display_cols if col.lower() != 'type']

        combined_data = []
        style_conditions = []

        # --- Process Original Row (Row Index 0) ---
        original_display = {'Row Type': 'Original', 'CF #': 'N/A', 'Changes': 'N/A'}
        for col in feature_cols:
             original_display[col] = original_row.get(col, 'N/A') # Use .get for safety
        combined_data.append(original_display)

        # Add style to distinguish the original row
        style_conditions.append({
            'if': {'row_index': 0},
            'fontWeight': 'bold',
            'backgroundColor': 'rgba(100, 100, 100, 0.15)' # Slightly different background
        })

        # --- Process Counterfactual Rows (Starting from Row Index 1) ---
        if not cf_rows.empty:
            for cf_idx, (row_label, cf_row) in enumerate(cf_rows.iterrows(), start=1):
                # combined_row_index corresponds to cf_idx since original is row 0
                cf_display = {'Row Type': f'CF {cf_idx}', 'CF #': cf_idx}
                changed_features_list = []

                for col in feature_cols:
                    original_val = original_row.get(col)
                    cf_val = cf_row.get(col)

                    # Check if the value changed
                    changed = False
                    try:
                        if pd.isna(original_val) and pd.isna(cf_val):
                            changed = False
                        elif pd.isna(original_val) or pd.isna(cf_val):
                            changed = True
                        elif isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                            if not np.isclose(original_val, cf_val, rtol=1e-05, atol=1e-08, equal_nan=True):
                                changed = True
                        elif original_val != cf_val:
                            changed = True
                    except TypeError:
                        if str(original_val) != str(cf_val):
                            changed = True

                    if changed:
                        cf_display[col] = cf_val
                        changed_features_list.append(col)
                        # Add style condition to highlight this changed cell
                        style_conditions.append({
                            'if': {'row_index': cf_idx, 'column_id': col},
                            'backgroundColor': '#3D9970', # Teal-ish highlight
                            'color': 'white',
                            'fontWeight': 'bold'
                        })
                    else:
                        # Indicate no change with em dash
                        cf_display[col] = "—"

                cf_display['Changes'] = ", ".join(changed_features_list) if changed_features_list else "None"
                combined_data.append(cf_display)

        # --- Define Columns for the Combined DataTable ---
        # Start with the special columns, then add the feature columns
        table_columns = [
            {"name": "Row Type", "id": "Row Type"},
            {"name": "CF #", "id": "CF #"},
            {"name": "Changed", "id": "Changes"}
        ] + [{"name": i, "id": i} for i in feature_cols]


        # --- Create the Combined DataTable ---
        combined_table = dash_table.DataTable(
            columns=table_columns,
            data=combined_data,
            style_table={'overflowX': 'auto', 'marginTop': '10px'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={ # Default cell style for all cells
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'border': '1px solid #555',
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '60px', 'width': '100px', 'maxWidth': '150px', # Adjust width
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            # Apply conditional styles (original row + changed cells)
            style_data_conditional=style_conditions,
            tooltip_data=[ # Show full value on hover
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in combined_data
            ],
            tooltip_duration=None # Keep tooltip visible
        )

        # Return a Div containing the combined table
        return html.Div([
            html.H5("Original vs. Counterfactuals:", style={'marginTop':'20px', 'fontWeight':'bold'}),
            combined_table
        ])

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: File not found - {os.path.basename(file_path)}")
        return P_component(f"Error: File not found - {os.path.basename(file_path)}", style={'color':'red'})
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: The file is empty - {os.path.basename(file_path)}")
        return P_component(f"Error: The file is empty - {os.path.basename(file_path)}", style={'color':'red'})
    except KeyError as e:
        print(f"Error processing CFE file {file_path}: Missing expected column {e}")
        traceback.print_exc()
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: Missing expected column {e} in {os.path.basename(file_path)}.")
        return P_component(f"Error: Missing expected column {e} in {os.path.basename(file_path)}.", style={'color': 'red'})
    except Exception as e:
        print(f"Error creating combined CFE table for {file_path}: {e}")
        traceback.print_exc()
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error processing counterfactuals file: {os.path.basename(file_path)} - {e}")
        return P_component(f"Error processing counterfactuals file: {os.path.basename(file_path)} - {e}", style={'color':'red'})


def register_job_page_callbacks(app):
    print("Registering job page callbacks...")
    
    theme_colors = {
        'background': '#0D3D66', 'header_background': '#1E3A5F',
        'content_background': '#104E78', 'status_background': '#145E88',
        'text_light': '#E0E0E0', 'text_medium': '#C0C0C0',
        'text_dark': '#FFFFFF', 'border_light': '#444'
    }

    # --- Callback to parse job name from URL ---
    @app.callback(
        Output('job-page-job-name-store', 'data'),
        Input('url', 'pathname') # Triggered when URL pathname changes (including initial load)
    )
    def update_job_store_from_url(pathname):
        """
        Parses the job name from the URL pathname (/job/job_name)
        and updates the job name store.
        """
        print(f"(URL Parser CB) Pathname received: {pathname}")
        if pathname and pathname.startswith('/job/'):
            try:
                job_name = pathname.split('/')[-1]
                if job_name:
                    print(f"(URL Parser CB) Extracted job name: {job_name}. Updating store.")
                    return job_name
                else:
                    print("(URL Parser CB) Job name empty after split.")
                    return None
            except Exception as e:
                print(f"(URL Parser CB) Error parsing pathname '{pathname}': {e}")
                return None
        else:
            print("(URL Parser CB) Pathname doesn't match expected '/job/...' format.")
            return None # Clear job name if path doesn't match

    @app.callback(
        Output('job-metadata-display', 'children'),
        Input('job-page-job-name-store', 'data')
    )
    def update_metadata_display(job_name):
        if not job_name:
            # Return a styled message consistent with theme
            return html.Div("Select a job to view metadata.", style={'color': theme_colors['text_medium'], 'padding': '10px'})

        print(f"(Metadata CB) Attempting to load metadata for job: {job_name}")

        # --- Construct the path to the metadata logfile ---
        metadata_filename = f"logfile"
        metadata_filepath = os.path.join(XAI_DIR, job_name, metadata_filename)
        print(f"(Metadata CB) Expecting metadata file at: {metadata_filepath}")
        # ----------------------------------------------------

        # --- Read metadata from the file ---
        metadata_json = None
        try:
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata_json = f.read()
            print(f"(Metadata CB) Successfully read metadata file: {metadata_filepath}")

        except FileNotFoundError:
            print(f"(Metadata CB) Metadata file not found: {metadata_filepath}")
            # Return a clear message if the file is missing
            return html.Div([
                html.Strong("Metadata file not found."),
                html.P(f"Expected location: {metadata_filepath}", style={'fontSize':'small', 'color':theme_colors['text_medium']})
            ], style={'color': 'orange', 'padding': '10px', 'border': f"1px solid {theme_colors['border_light']}", 'borderRadius':'5px', 'backgroundColor':'rgba(255, 165, 0, 0.1)'}) # Orange theme for warning
        except Exception as e:
            print(f"(Metadata CB) Error reading metadata file {metadata_filepath}: {e}")
            traceback.print_exc()
            return html.Div(f"Error reading metadata file: {e}", style={'color': 'red', 'padding': '10px'}) # Red theme for error
        # -----------------------------------
        
        # --- If file read successfully, proceed with parsing and display ---
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                print(f"(Metadata CB) Successfully parsed JSON metadata for {job_name}")

                # --- Build Display Components (using helpers defined earlier) ---
                display_elements = []

                # 1. General Job Info
                job_info = {
                    "Run Timestamp (UTC)": metadata.get("run_timestamp_utc"),
                    "Status": metadata.get("status"),
                    "Dataset Path": metadata.get("dataset_path"),
                    "Model Name": metadata.get("model_name"),
                    "Label Column": metadata.get("label_column_used"),
                    "Sequence Length": metadata.get("sequence_length")
                }
                display_elements.append(create_info_section("Job Summary", {k: v for k, v in job_info.items() if v is not None}, theme_colors, format_floats=False))

                # 2. Data Summary
                data_summary = {
                    "Total Rows": metadata.get("data_total_rows"),
                    "Training Rows": metadata.get("data_training_rows"),
                    "Testing Rows": metadata.get("data_testing_rows"),
                    "Features": metadata.get("data_num_features"),
                    "Anomalies (Ground Truth)": metadata.get("data_num_anomalies_ground_truth"),
                    "Anomalies (Predicted)": metadata.get("data_num_anomalies_predicted")
                }
                data_summary_filtered = {k: v for k, v in data_summary.items() if v is not None}
                if data_summary_filtered:
                     display_elements.append(create_info_section("Data Summary", data_summary_filtered, theme_colors, format_floats=False))

                # 3. Performance Metrics
                metrics = metadata.get("evaluation_metrics", {})
                if metrics:
                    display_elements.append(create_info_section("Performance Metrics", metrics, theme_colors))

                # 4. Execution Times
                exec_times = {
                    "Total (s)": metadata.get("execution_time_total_seconds"),
                    "Simulation (s)": metadata.get("execution_time_simulation_seconds"),
                    "Training (s)": metadata.get("execution_time_training_seconds"),
                    "Detection (s)": metadata.get("execution_time_detection_seconds"),
                    "XAI (s)": metadata.get("execution_time_xai_seconds")
                }
                exec_times_filtered = {k: v for k, v in exec_times.items() if v is not None}
                if exec_times_filtered:
                    display_elements.append(create_info_section("Execution Times", exec_times_filtered, theme_colors))

                # 5. Model Parameters (collapsible)
                model_params = metadata.get("model_params")
                if model_params:
                    display_elements.append(html.Div([
                         html.Details([
                             html.Summary("Model Parameters", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                             create_pretty_dict_list_display(model_params)
                         ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))

                # 6. XAI Settings (collapsible)
                xai_settings = metadata.get("xai_settings")
                if xai_settings:
                     display_elements.append(html.Div([
                         html.Details([
                             html.Summary("XAI Settings", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                            create_pretty_dict_list_display(xai_settings)
                         ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))

                # 7. Anomaly Injection Params (collapsible, if any)
                injection_params = metadata.get("anomaly_injection_params")
                if injection_params: # Check if list is not empty or None
                    display_elements.append(html.Div([
                        html.Details([
                            html.Summary("Anomaly Injection Parameters", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                            create_pretty_dict_list_display(injection_params)
                        ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))

                # --- Create the Grid Container ---
                grid_container = html.Div(
                    children=display_elements, # Place all section divs inside the container
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(350px, 1fr))',
                        'gap': '20px', # Space between grid items (rows and columns)
                        'width': '100%' # Ensure container takes full width
                    }
                )
                
                print(f"(Metadata CB) Successfully generated display components for {job_name} from file.")
                return grid_container 

            except json.JSONDecodeError as e:
                print(f"(Metadata CB) Error decoding metadata JSON from file {metadata_filepath}: {e}")
                return html.Div(f"Error loading metadata: Invalid JSON format in {metadata_filename} - {e}", style={'color': 'red', 'padding': '10px'})
            except Exception as e:
                print(f"(Metadata CB) Error generating metadata display for {job_name} from file: {e}")
                traceback.print_exc()
                return html.Div(f"An error occurred while displaying metadata: {e}", style={'color': 'red', 'padding': '10px'})
        else:
            # This case should theoretically not be reached if file reading failed earlier,
            # but included for completeness.
             return html.Div("Failed to load metadata content.", style={'color': 'red', 'padding': '10px'})

    # --- Callback 1: Fetch and Store Data ---
    @app.callback(
        [
            Output('job-page-data-store', 'data'),
            Output('job-status-display', 'children'),
            Output('loading-output-jobpage', 'children') # Controls the loading indicator text/presence
        ],
        [
            Input('job-page-job-name-store', 'data'),      # Trigger on job change
            Input('job-page-interval-component', 'n_intervals') # Trigger on interval
        ],
    )
    def update_data_store(job_name, n_intervals):
        """
        Fetches data based on triggers:
        - Always fetches when job_name changes (initial load).
        - Fetches periodically ONLY if job_name starts with 'job_stream_'.
        - Stores fetched data in dcc.Store. Updates status.
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'
        current_time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        display_name = get_display_job_name(job_name) # Get display name

        if not job_name:
            return None, f"No job selected. Last checked: {current_time_display}", None # Clear data, update status, clear loading

        is_streaming_job = job_name.startswith("job_stream_")
        is_batch_job = job_name.startswith("job_batch_")
        triggered_by_job_change = trigger_id == 'job-page-job-name-store' or trigger_id == 'initial load'
        triggered_by_interval = trigger_id == 'job-page-interval-component'

        should_fetch = False
        fetch_reason = ""
        start_time_iso = None

        if triggered_by_job_change:
            if is_streaming_job or is_batch_job:
                should_fetch = True
                fetch_reason = f"Job selected/changed to '{display_name}'"
                if is_batch_job:
                    start_time_iso = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
                    print(f"(Data Fetch CB) Using epoch start for batch job '{display_name}'.") 
                else: # is_streaming_job
                    lookback_minutes = 60
                    start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                    start_time_iso = start_time.isoformat()
                    print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming job '{display_name}' initial load.") 
            else:
                print(f"(Data Fetch CB) Job changed to unrecognized type: {job_name}. No fetch.")

        elif triggered_by_interval:
            if is_streaming_job:
                should_fetch = True
                fetch_reason = f"Interval trigger for streaming job '{display_name}'" 
                lookback_minutes = 10 # Fetch recent data
                start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                start_time_iso = start_time.isoformat()
                print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming interval fetch.")
            elif is_batch_job:
                print(f"(Data Fetch CB) Interval trigger ignored for batch job '{display_name}'.") 
                status_msg = f"Data loaded for batch job: '{display_name}'. Last Update: {current_time_display} UTC" 
                return dash.no_update, status_msg, dash.no_update # Only update status
            else:
                print(f"(Data Fetch CB) Interval trigger for unrecognized job type: {job_name}. No fetch.")

        if should_fetch and start_time_iso:
            print(f"(Data Fetch CB) Condition met: {fetch_reason}. Fetching data from {start_time_iso}...")
            status_msg = f"Fetching data for job '{display_name}' ({'Streaming' if is_streaming_job else 'Batch'})..." 
            data_json = None
            try:
                handler = get_handler()
                df = handler.handle_get_data(timestamp=start_time_iso, job_name=job_name) 

                if df is not None and not df.empty:
                    print(f"(Data Fetch CB) Successfully fetched data. Shape: {df.shape}")
                    data_json = df.to_json(date_format='iso', orient='split')
                    status_msg = f"Data updated for job '{display_name}'. Reason: {fetch_reason}. Records: {len(df)}. Timestamp: {current_time_display}" 
                else:
                    print(f"(Data Fetch CB) Received empty DataFrame or None for job '{display_name}'.") 
                    status_msg = f"No new data found for job '{display_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}" 
                    data_json = None

            except Exception as e:
                print(f"(Data Fetch CB) Error fetching data for job '{display_name}':") 
                traceback.print_exc()
                status_msg = f"Error fetching data for job '{display_name}': {e}. Timestamp: {current_time_display}" 

            return data_json, status_msg, None

        else:
            print("(Data Fetch CB) No fetch condition met or start_time_iso not set. Returning no_update.")
            return dash.no_update, dash.no_update, dash.no_update


    # --- Callback 2: Update Graph from Stored Data ---
    @app.callback(
        Output('timeseries-anomaly-graph', 'figure'),
        Input('job-page-data-store', 'data'),
        State('job-page-job-name-store', 'data')
    )
    def update_graph_from_data(stored_data_json, job_name):
        """
        Updates the graph based on the data stored in dcc.Store.
        All numeric columns are added as traces, hidden by default ('legendonly').
        Anomaly markers are plotted visibly.
        """
        job_title_name = get_display_job_name(job_name) if job_name else "No Job Selected" 
        fig = go.Figure(layout={'template': 'plotly_dark', 'title': f'Loading Data for {job_title_name}...' })

        if stored_data_json is None:
            print("(Graph Update CB) No data in store.")
            fig.update_layout(title=f'No Data Available for {job_title_name}', xaxis={'visible': False}, yaxis={'visible': False})
            return fig

        print("(Graph Update CB) Data found in store, processing graph...")
        try:
            df = pd.read_json(StringIO(stored_data_json), orient='split')
            print(f"(Graph Update CB) DataFrame loaded. Shape: {df.shape}")

            # --- Convert Timestamp ---
            x_axis_data = df.index # Default
            if 'timestamp' in df.columns:
                # Your timestamp conversion logic seems fine, adapt if needed
                epoch_start = pd.Timestamp('1970-01-01 00:00:00', tzinfo=timezone.utc)
                relative_seconds = pd.to_numeric(df['timestamp'], errors='coerce')
                df['datetime'] = epoch_start + pd.to_timedelta(relative_seconds, unit='s')
                x_axis_data = df['datetime']
                print(f"(Graph Update CB) Converted timestamp to datetime column 'datetime'.")
            else:
                print("(Graph Update CB) Warning: 'timestamp' column not found. Using DataFrame index for X-axis.")

            # --- Identify Columns ---
            cols_to_exclude = {'timestamp', 'datetime', 'label', 'is_anomaly', 'injected_anomaly'}
            numeric_cols = df.select_dtypes(include=['number']).columns
            available_y_cols = [col for col in numeric_cols if col not in cols_to_exclude]
            print(f"(Graph Update CB) Found {len(available_y_cols)} numeric columns available for plotting: {available_y_cols}")

            # --- Create Graph ---
            fig = go.Figure(layout=go.Layout(
                title=f"Time Series Data: {job_title_name}", 
                template="plotly_dark",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                legend_title_text='Features',
                uirevision=job_name # Use original job_name for uirevision
            ))

            # --- Add Traces ---
            if not available_y_cols:
                 print("(Graph Update CB) No numeric Y-axis columns found to plot.")
                 fig.update_layout(title=f"No Plottable Numeric Data Found for {job_title_name}") 
            else:
                print(f"(Graph Update CB) Adding {len(available_y_cols)} traces with 'legendonly' visibility.")
                for col_name in available_y_cols:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                         fig.add_trace(go.Scattergl(x=x_axis_data, y=df[col_name], mode='lines', name=col_name, visible='legendonly'))
                    else:
                         print(f"(Graph Update CB) Warning: Column '{col_name}' was in available list but is not numeric. Skipping.")

            # --- Add Anomaly Markers ---
            anomaly_col = 'is_anomaly'
            if anomaly_col in df.columns:
                anomalies_df = df[df[anomaly_col] == 1]
                print(f"(Graph Update CB) Found {len(anomalies_df)} anomalies indicated by column '{anomaly_col}'.")
                if not anomalies_df.empty:
                    y_anomaly_col = available_y_cols[0] if available_y_cols else None
                    y_values_for_anomalies = anomalies_df[y_anomaly_col] if y_anomaly_col else 0
                    anomaly_x_data = anomalies_df['datetime'] if 'datetime' in anomalies_df else anomalies_df.index
                    fig.add_trace(go.Scattergl(
                        x=anomaly_x_data, y=y_values_for_anomalies,
                        mode='markers', name='Detected Anomaly',
                        marker=dict(color='red', size=8, symbol='x'),
                        visible='legendonly' # Changed to legendonly as requested elsewhere
                    ))
                else:
                    print("(Graph Update CB) Anomaly column found, but no anomalies detected (value != 1).")
            else:
                print(f"(Graph Update CB) Anomaly indicator column '{anomaly_col}' not found in data.")

        except Exception as e:
            print(f"(Graph Update CB) Error processing stored data or plotting:")
            traceback.print_exc()
            fig = go.Figure(layout={
                'template': 'plotly_dark', 'title': f'Error Displaying Data for {job_title_name}', 
                'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                'annotations': [{'text': f'An error occurred: {str(e)}', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
            })
            sys.stdout.flush()

        sys.stdout.flush() # Ensure prints are flushed
        return fig # Return only the figure
    
    @app.callback(
        [Output('xai-results-content', 'children'),
         Output('xai-results-section', 'style')], # To hide/show the whole section
        [Input('job-page-job-name-store', 'data')] # Trigger when job name changes
    )
    def update_xai_display(job_name):
        """
        Scans the XAI directory for the current job, identifies subdirectories
        named after XAI methods, and displays their contents (images, html, csv).
        """
        sys.stdout.flush()

        if not job_name:
            return "No job selected.", {'display': 'none'} # Hide section if no job

        print(f"(XAI Display CB) Checking results for job: {job_name}")
        # Use the corrected XAI_DIR variable
        job_xai_base_path = os.path.join(XAI_DIR, job_name)
        xai_content_blocks = [] # List to hold Divs for each method
        found_any_results = False

        # Supported file extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
        html_extension = '.html'
        csv_extension = '.csv'

        # Check if the base job directory exists
        if not os.path.isdir(job_xai_base_path):
            print(f"(XAI Display CB) Base XAI directory not found: {job_xai_base_path}")
            # Return message, show the section to display the message
            return f"No XAI results found for job '{job_name}'. Base directory missing: {job_xai_base_path}", {'display': 'block'}

        try:
            # List potential XAI method subdirectories
            # Define known/expected method names if possible for better filtering
            # known_methods = {'ShapExplainer', 'LimeExplainer', 'DiceExplainer'}
            method_subdirs = []
            for item in os.listdir(job_xai_base_path):
                item_path = os.path.join(job_xai_base_path, item)
                if os.path.isdir(item_path):
                     # Optional: Check if 'item' name matches known_methods if you have a list
                     method_subdirs.append(item) # Assume all subdirs are methods for now

            if not method_subdirs:
                print(f"(XAI Display CB) No subdirectories found within: {job_xai_base_path}")
                return f"No XAI method subdirectories found within '{job_xai_base_path}'.", {'display': 'block'}

            # Process each found method subdirectory
            for method_name in sorted(method_subdirs): # Sort for consistent order
                method_path = os.path.join(job_xai_base_path, method_name)
                method_file_components = [] # List to hold components for files within this method
                print(f"(XAI Display CB) Scanning method directory: {method_path}")

                try:
                    # List and process files within the method directory
                    files_in_method_dir = [f for f in os.listdir(method_path) if os.path.isfile(os.path.join(method_path, f))]

                    if not files_in_method_dir:
                         print(f"  - No files found in {method_path}")
                         method_file_components.append(html.P("(No displayable files found in this directory)", style={'color':'#aaa'}))
                    else:
                        for filename in sorted(files_in_method_dir): # Sort files
                            file_path = os.path.join(method_path, filename)
                            _, extension = os.path.splitext(filename.lower())

                            # Generate the correct URL using the helper
                            asset_url = get_asset_url(job_name, method_name, filename)

                            component_to_add = None
                            if extension in image_extensions:
                                print(f"  - Found Image: {filename}, URL: {asset_url}")
                                component_to_add = html.Img(src=asset_url,
                                                             alt=f"{method_name} - {filename}",
                                                             style={'maxWidth': '95%', 'height': 'auto', 'marginTop': '10px', 'border':'1px solid #444', 'display':'block', 'marginLeft':'auto', 'marginRight':'auto'}) # Center images
                                print(component_to_add)
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file
                            
                            elif extension == html_extension:
                                print(f"  - Found HTML: {filename}, URL: {asset_url}")
                                component_to_add = html.Iframe(src=asset_url,
                                                               style={'width': '100%', 'height': '300px', 'marginTop': '10px', 'border': '1px solid #444', 'backgroundColor': 'white'})
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file
                            
                            elif extension == csv_extension:
                                print(f"  - Found CSV: {filename}")
                                # Use the helper function to create a DataTable
                                component_to_add = create_cfe_delta_table(file_path)
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file

                            sys.stdout.flush()

                except Exception as e:
                    print(f"(XAI Display CB) Error scanning/processing files in directory {method_path}: {e}")
                    method_file_components.append(html.P(f"Error processing results for {method_name}: {e}", style={'color':'red'}))

                print(f"method_file_components (after last append): {method_file_components}")
                # Create a block for this method if it has any components (even error messages)
                if method_file_components:
                    xai_content_blocks.append(html.Div([
                        # Method Title
                        html.H4(f"{method_name} Results", style={'borderBottom': '1px solid #555', 'paddingBottom': '5px', 'marginTop': '25px', 'marginBottom': '15px', 'color':'#eee'}),
                        # File Components
                        *method_file_components # Unpack the list of components
                    ], className="xai-method-block", style={'marginBottom': '30px', 'padding': '15px', 'border': '1px solid #555', 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.5)'})) # Style the block

            # --- Final Check and Return ---
            if not found_any_results:
                # This message means subdirectories were found, but no displayable files inside them
                 print("(XAI Display CB) Method subdirectories found, but no displayable files within them.")
                 return f"No displayable XAI results files (.png, .html, .csv) found for job '{job_name}' in method subdirectories.", {'display': 'block'}

            print("(XAI Display CB) Finished processing XAI results. Returning content blocks.")
            # Return the list of method blocks, ensure section is visible
            return xai_content_blocks, {'display': 'block'}

        except Exception as e:
            print(f"(XAI Display CB) General error processing XAI for job {job_name}:")
            traceback.print_exc()
            return f"An error occurred while trying to load XAI results: {e}", {'display': 'block'}

    # --- Callback For The "Back To Home" Button ---
    @app.callback(
        Output('url', 'pathname'),            # Target the 'pathname' of dcc.Location(id='url')
        Input('back-to-home-button', 'n_clicks'), # Listen to button clicks
        prevent_initial_call=True             # IMPORTANT: Don't run when the page loads
    )
    def go_back_to_home(n_clicks):
        if n_clicks and n_clicks > 0:
            print("Back button clicked, navigating to home ('/')")
            # NOTE: Ensure 'dcc.Location(id="url")' exists in your main app layout (app.py)
            return "/"  # Return the path for the home page
        return dash.no_update # If no clicks (or initial call), do nothing
    # --- End Callback For "Back To Home" Button ---

    # --- Add other callbacks if needed ---

    print("Job page callbacks registered.")
    sys.stdout.flush() # Ensure registration message is flushed