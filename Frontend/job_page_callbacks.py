# job_page_callbacks.py
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
from io import StringIO # Needed to read JSON from Store

from get_handler import get_handler # Assuming this helper exists

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
    """Reads a CFE CSV, calculates differences, and returns a styled DataTable."""
    try:
        df_cfe = pd.read_csv(file_path)

        # Find the original row
        original_rows = df_cfe[df_cfe['type'].str.lower() == 'original']
        if original_rows.empty:
            return html.P(f"Error: 'original' row not found in {os.path.basename(file_path)}.", style={'color': 'red'})
        original_row = original_rows.iloc[0]

        # Get counterfactual rows
        cf_rows = df_cfe[df_cfe['type'].str.lower() == 'counterfactual']
        if cf_rows.empty:
            return html.P(f"No 'counterfactual' rows found in {os.path.basename(file_path)}.", style={'color': 'orange'})

        # Identify feature columns (exclude 'type')
        feature_cols = [col for col in df_cfe.columns if col.lower() != 'type']

        display_data = []
        style_conditions = []

        # Process each counterfactual row
        for idx, cf_row in cf_rows.iterrows():
            display_row = {'Counterfactual #': idx - original_row.name} # Simple row number
            changed_features_list = []
            row_index = idx - original_row.name - 1 # 0-based index for styling

            for col in feature_cols:
                original_val = original_row[col]
                cf_val = cf_row[col]

                # Check if the value changed (handle potential type differences and NaN)
                changed = False
                # Try numeric comparison first with tolerance for floats
                try:
                    # Check for NaN equality explicitly
                    if pd.isna(original_val) and pd.isna(cf_val):
                        changed = False
                    elif pd.isna(original_val) or pd.isna(cf_val):
                        changed = True # One is NaN, the other isn't
                    # Use isclose for floats, direct compare otherwise
                    elif isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                         if not np.isclose(original_val, cf_val, rtol=1e-05, atol=1e-08):
                              changed = True
                    elif original_val != cf_val: # Direct comparison for others (strings etc)
                        changed = True
                except TypeError: # Fallback if types are incompatible for comparison
                     if str(original_val) != str(cf_val):
                           changed = True

                if changed:
                    # Show the new counterfactual value
                    display_row[col] = cf_val
                    changed_features_list.append(col)
                    # Add style condition to highlight this cell
                    style_conditions.append({
                        'if': {'row_index': row_index, 'column_id': col},
                        'backgroundColor': '#3D9970', # Teal-ish highlight
                        'color': 'white',
                        'fontWeight': 'bold'
                    })
                else:
                    # Indicate no change (e.g., with a dash)
                    display_row[col] = "—" # Em dash for clarity

            display_row['Changes'] = ", ".join(changed_features_list) if changed_features_list else "None"
            display_data.append(display_row)

        # Define columns for the DataTable
        table_columns = [{"name": "CF #", "id": "Counterfactual #"}] + \
                        [{"name": "Changed Features", "id": "Changes"}] + \
                        [{"name": i, "id": i} for i in feature_cols]

        delta_table = dash_table.DataTable(
            columns=table_columns,
            data=display_data,
            style_table={'overflowX': 'auto', 'marginTop': '10px'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={ # Default cell style
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'border': '1px solid #555',
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '80px', 'width': '120px', 'maxWidth': '180px', # Adjust width as needed
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            # Apply conditional styles LAST to override defaults
            style_data_conditional=style_conditions,
            tooltip_data=[ # Show full value on hover for truncated cells
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in display_data
            ],
             tooltip_duration=None # Keep tooltip visible
        )

        return html.Div([
            html.P("Counterfactual Explanations (Highlighted cells show changed values from original):", style={'marginTop':'10px', 'fontWeight':'bold'}),
            delta_table
            ])

    except Exception as e:
        print(f"Error creating CFE delta table for {file_path}: {e}")
        traceback.print_exc()
        return html.P(f"Error processing counterfactuals file: {os.path.basename(file_path)} - {e}", style={'color':'red'})
# --------------------------------------------------------

def register_job_page_callbacks(app):
    print("Registering job page callbacks...")

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

    # --- Callback 1: Fetch and Store Data ---
    # (Keep this callback exactly as it was in your provided code)
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

        if not job_name:
            return None, f"No job selected. Last checked: {current_time_display}", None # Clear data, update status, clear loading

        is_streaming_job = job_name.startswith("job_stream_")
        is_batch_job = job_name.startswith("job_batch_")
        # Check if the trigger is the job store update OR if it's the first load (no specific trigger yet)
        triggered_by_job_change = trigger_id == 'job-page-job-name-store' or trigger_id == 'initial load'
        triggered_by_interval = trigger_id == 'job-page-interval-component'

        should_fetch = False
        fetch_reason = ""
        start_time_iso = None

        if triggered_by_job_change:
            if is_streaming_job or is_batch_job:
                should_fetch = True
                fetch_reason = f"Job selected/changed to '{job_name}'"
                if is_batch_job:
                    start_time_iso = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
                    print(f"(Data Fetch CB) Using epoch start for batch job '{job_name}'.")
                else: # is_streaming_job
                    lookback_minutes = 60
                    start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                    start_time_iso = start_time.isoformat()
                    print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming job '{job_name}' initial load.")
            else:
                print(f"(Data Fetch CB) Job changed to unrecognized type: {job_name}. No fetch.")

        elif triggered_by_interval:
            if is_streaming_job:
                should_fetch = True
                fetch_reason = f"Interval trigger for streaming job '{job_name}'"
                lookback_minutes = 10 # Fetch recent data
                start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                start_time_iso = start_time.isoformat()
                print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming interval fetch.")
            elif is_batch_job:
                print(f"(Data Fetch CB) Interval trigger ignored for batch job '{job_name}'.")
                status_msg = f"Data loaded for batch job '{job_name}'. (No periodic update). Last checked: {current_time_display}"
                return dash.no_update, status_msg, dash.no_update # Only update status
            else:
                print(f"(Data Fetch CB) Interval trigger for unrecognized job type: {job_name}. No fetch.")

        if should_fetch and start_time_iso:
            print(f"(Data Fetch CB) Condition met: {fetch_reason}. Fetching data from {start_time_iso}...")
            status_msg = f"Fetching data for job '{job_name}' ({'Streaming' if is_streaming_job else 'Batch'})..."
            data_json = None
            try:
                handler = get_handler()
                df = handler.handle_get_data(timestamp=start_time_iso, job_name=job_name)

                if df is not None and not df.empty: # Check df is not None
                    print(f"(Data Fetch CB) Successfully fetched data. Shape: {df.shape}")
                    # --- Data Merging Logic (Example: Append and drop duplicates) ---
                    # If you want to accumulate data for streaming jobs:
                    # 1. Get existing data from store (need Store component as State)
                    # 2. Parse existing JSON to DataFrame
                    # 3. Concatenate old and new DataFrames
                    # 4. Drop duplicates based on index/timestamp
                    # 5. Sort by index/timestamp
                    # 6. Convert back to JSON
                    # For simplicity, current code replaces data on each fetch.
                    # Add merging logic here if needed, especially for streaming.
                    data_json = df.to_json(date_format='iso', orient='split')
                    status_msg = f"Data updated for job '{job_name}'. Reason: {fetch_reason}. Records: {len(df)}. Timestamp: {current_time_display}"
                else:
                    print(f"(Data Fetch CB) Received empty DataFrame or None for job '{job_name}'.")
                    status_msg = f"No new data found for job '{job_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}"
                    # Decide: return no_update to keep old data, or None to clear?
                    # Clearing seems reasonable if the fetch returns nothing.
                    data_json = None # Clear graph if no data received

            except Exception as e:
                print(f"(Data Fetch CB) Error fetching data for job '{job_name}':")
                traceback.print_exc()
                status_msg = f"Error fetching data for job '{job_name}': {e}. Timestamp: {current_time_display}"
                data_json = None # Clear data on error

            # Return data (or None), status, and None for loading text (clears it)
            return data_json, status_msg, None

        else:
            # No fetch condition met, return no_update for all outputs
            print("(Data Fetch CB) No fetch condition met or start_time_iso not set. Returning no_update.")
            return dash.no_update, dash.no_update, dash.no_update


    # --- Callback 2: Update Graph from Stored Data ---
    # MODIFIED: Removed dropdown Input/Output, adjusted logic
    @app.callback(
        Output('timeseries-anomaly-graph', 'figure'), # Only outputting the figure now
        Input('job-page-data-store', 'data'), # Trigger when stored data changes
        State('job-page-job-name-store', 'data') # Get job name for title etc.
        # REMOVED: Input('y-axis-dropdown', 'value')
        # REMOVED: Output('y-axis-dropdown', 'options')
    )
    def update_graph_from_data(stored_data_json, job_name):
        """
        Updates the graph based on the data stored in dcc.Store.
        All numeric columns are added as traces, hidden by default ('legendonly').
        Anomaly markers are plotted visibly.
        """
        # REMOVED: selected_y_columns from arguments

        job_title_name = job_name if job_name else "No Job Selected"
        # Default empty figure
        fig = go.Figure(layout={'template': 'plotly_dark', 'title': f'Loading Data for {job_title_name}...' })

        if stored_data_json is None:
            print("(Graph Update CB) No data in store.")
            fig.update_layout(title=f'No Data Available for {job_title_name}', xaxis={'visible': False}, yaxis={'visible': False})
            return fig # Return empty figure

        print("(Graph Update CB) Data found in store, processing graph...")
        try:
            # --- Read DataFrame from stored JSON ---
            # Use StringIO to read the JSON string as if it were a file
            df = pd.read_json(StringIO(stored_data_json), orient='split')
            print(f"(Graph Update CB) DataFrame loaded. Shape: {df.shape}")

            # --- IMPORTANT: Convert Timestamp Column ---
            # Assuming 'timestamp' column exists and contains seconds since epoch
            if 'timestamp' in df.columns:
                epoch_start = pd.Timestamp('1970-01-01 00:00:00', tzinfo=timezone.utc)
                relative_seconds = pd.to_numeric(df['timestamp'], errors='coerce')
                df['datetime'] = epoch_start + pd.to_timedelta(relative_seconds, unit='s') # Create a new datetime column
                # Consider setting datetime as index IF timestamps are unique and sorted
                # df = df.set_index('datetime').sort_index()
                # If using index, x_axis_data = df.index
                # If not using index, x_axis_data = df['datetime']
                x_axis_data = df['datetime'] # Use the converted datetime column for X-axis
                print(f"(Graph Update CB) Converted timestamp to datetime column 'datetime'.")
            else:
                # Fallback or error if no timestamp? Using index perhaps?
                print("(Graph Update CB) Warning: 'timestamp' column not found. Using DataFrame index for X-axis.")
                x_axis_data = df.index # Use index if timestamp is missing

            # --- Identify Columns to Plot ---
            cols_to_exclude = {'timestamp', 'datetime', 'label', 'is_anomaly', 'injected_anomaly'} # Exclude original timestamp and new datetime
            # Select numeric columns first
            numeric_cols = df.select_dtypes(include=['number']).columns
            # Filter out excluded columns
            available_y_cols = [col for col in numeric_cols if col not in cols_to_exclude]
            print(f"(Graph Update CB) Found {len(available_y_cols)} numeric columns available for plotting: {available_y_cols}")

            # --- Create Graph ---
            fig = go.Figure(layout=go.Layout(
                title=f"Time Series Data for {job_title_name} (Click Legend to Show/Hide)",
                template="plotly_dark",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                legend_title_text='Features',
                uirevision=job_name # Preserve zoom/pan when data updates for the same job
            ))

            # --- Add Traces for Available Columns (Hidden by Default) ---
            if not available_y_cols:
                 print("(Graph Update CB) No numeric Y-axis columns found to plot.")
                 fig.update_layout(title=f"No Plottable Numeric Data Found for {job_title_name}")
            else:
                print(f"(Graph Update CB) Adding {len(available_y_cols)} traces with 'legendonly' visibility.")
                for col_name in available_y_cols:
                    # Check again just in case (though select_dtypes should handle it)
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                         fig.add_trace(go.Scattergl( # Use Scattergl for potentially better performance
                             x=x_axis_data,
                             y=df[col_name],
                             mode='lines',
                             name=col_name,
                             visible='legendonly' # KEY CHANGE: Show in legend, hide on graph
                         ))
                    else:
                         print(f"(Graph Update CB) Warning: Column '{col_name}' was in available list but is not numeric. Skipping.")

            # --- Add Anomaly Markers (Visible by Default) ---
            anomaly_col = 'is_anomaly' # Or your anomaly indicator column
            if anomaly_col in df.columns:
                anomalies_df = df[df[anomaly_col] == 1]
                print(f"(Graph Update CB) Found {len(anomalies_df)} anomalies indicated by column '{anomaly_col}'.")
                if not anomalies_df.empty:
                    # Determine Y value for anomalies. Plot against the first available numeric column? Or 0?
                    # Let's plot against the first available column for context.
                    y_anomaly_col = available_y_cols[0] if available_y_cols else None
                    if y_anomaly_col:
                        print(f"(Graph Update CB) Plotting anomaly markers against column '{y_anomaly_col}'.")
                        y_values_for_anomalies = anomalies_df[y_anomaly_col]
                    else:
                         # Fallback if no numeric columns exist (edge case)
                        print("(Graph Update CB) No suitable column for anomaly Y values, plotting at Y=0.")
                        y_values_for_anomalies = 0 # Plot at y=0 if no other columns

                    fig.add_trace(go.Scattergl(
                        x=anomalies_df['datetime'] if 'datetime' in anomalies_df else anomalies_df.index, # Use datetime if available
                        y=y_values_for_anomalies,
                        mode='markers',
                        name='Detected Anomaly',
                        marker=dict(color='red', size=8, symbol='x'),
                        visible=True # Anomalies are visible
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
            sys.stdout.flush() # Ensure error prints are flushed

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

                            print(f"method_file_components (after each file): {method_file_components}")
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

    # --- Add other callbacks if needed ---

    print("Job page callbacks registered.")
    sys.stdout.flush() # Ensure registration message is flushed