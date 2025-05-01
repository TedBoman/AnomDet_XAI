# job_page_callbacks.py
import sys
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import pandas as pd
import traceback
from datetime import datetime, timedelta, timezone
from io import StringIO # Needed to read JSON from Store

# <<< Make sure this import points correctly to where get_display_job_name is defined >>>
# If job_page.py is in a 'pages' subfolder:
from pages.job_page import get_display_job_name
# If job_page.py is in the same folder:
# from job_page import get_display_job_name # Adjust if necessary

from get_handler import get_handler # Assuming this helper exists

def register_job_page_callbacks(app):
    print("Registering job page callbacks...")

    # --- Callback to parse job name from URL (Your existing code) ---
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


    # --- Callback 1: Fetch and Store Data (Your existing code - uses get_display_job_name) ---
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
                fetch_reason = f"Job selected/changed to '{display_name}'" # Use display_name
                if is_batch_job:
                    start_time_iso = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
                    print(f"(Data Fetch CB) Using epoch start for batch job '{display_name}'.") # Use display_name
                else: # is_streaming_job
                    lookback_minutes = 60
                    start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                    start_time_iso = start_time.isoformat()
                    print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming job '{display_name}' initial load.") # Use display_name
            else:
                print(f"(Data Fetch CB) Job changed to unrecognized type: {job_name}. No fetch.")

        elif triggered_by_interval:
            if is_streaming_job:
                should_fetch = True
                fetch_reason = f"Interval trigger for streaming job '{display_name}'" # Use display_name
                lookback_minutes = 10 # Fetch recent data
                start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                start_time_iso = start_time.isoformat()
                print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming interval fetch.")
            elif is_batch_job:
                print(f"(Data Fetch CB) Interval trigger ignored for batch job '{display_name}'.") # Use display_name
                # Assuming you want the <br> fix applied here too eventually
                status_msg = f"Data loaded for batch job: '{display_name}'. Last Update: {current_time_display} UTC" # Use display_name
                # If using html.Br() list approach:
                # status_content = [f"Data loaded for batch job: '{display_name}'.", html.Br(), f"Last Update: {current_time_display} UTC"]
                # return dash.no_update, status_content, dash.no_update
                return dash.no_update, status_msg, dash.no_update # Only update status
            else:
                print(f"(Data Fetch CB) Interval trigger for unrecognized job type: {job_name}. No fetch.")

        if should_fetch and start_time_iso:
            print(f"(Data Fetch CB) Condition met: {fetch_reason}. Fetching data from {start_time_iso}...")
            status_msg = f"Fetching data for job '{display_name}' ({'Streaming' if is_streaming_job else 'Batch'})..." # Use display_name
            data_json = None
            try:
                handler = get_handler()
                df = handler.handle_get_data(timestamp=start_time_iso, job_name=job_name) # Use original job_name for backend

                if df is not None and not df.empty:
                    print(f"(Data Fetch CB) Successfully fetched data. Shape: {df.shape}")
                    data_json = df.to_json(date_format='iso', orient='split')
                    status_msg = f"Data updated for job '{display_name}'. Reason: {fetch_reason}. Records: {len(df)}. Timestamp: {current_time_display}" # Use display_name
                else:
                    print(f"(Data Fetch CB) Received empty DataFrame or None for job '{display_name}'.") # Use display_name
                    status_msg = f"No new data found for job '{display_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}" # Use display_name
                    data_json = None

            except Exception as e:
                print(f"(Data Fetch CB) Error fetching data for job '{display_name}':") # Use display_name
                traceback.print_exc()
                status_msg = f"Error fetching data for job '{display_name}': {e}. Timestamp: {current_time_display}" # Use display_name
                data_json = None

            return data_json, status_msg, None

        else:
            print("(Data Fetch CB) No fetch condition met or start_time_iso not set. Returning no_update.")
            return dash.no_update, dash.no_update, dash.no_update


    # --- Callback 2: Update Graph from Stored Data (Your existing code - uses get_display_job_name) ---
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
        job_title_name = get_display_job_name(job_name) if job_name else "No Job Selected" # Use helper
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
                title=f"Time Series Data: {job_title_name}", # Use display name
                template="plotly_dark",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                legend_title_text='Features',
                uirevision=job_name # Use original job_name for uirevision
            ))

            # --- Add Traces ---
            if not available_y_cols:
                 print("(Graph Update CB) No numeric Y-axis columns found to plot.")
                 fig.update_layout(title=f"No Plottable Numeric Data Found for {job_title_name}") # Use display name
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
                'template': 'plotly_dark', 'title': f'Error Displaying Data for {job_title_name}', # Use display name
                'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                'annotations': [{'text': f'An error occurred: {str(e)}', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
            })
            sys.stdout.flush()

        sys.stdout.flush()
        return fig

    # --- <<< ADDED CALLBACK FOR THE 'BACK TO HOME' BUTTON >>> ---
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
    # --- <<< END 'BACK TO HOME' CALLBACK >>> ---

    # --- Add other callbacks for XAI or other plots if needed ---

    print("Job page callbacks registered.")
    sys.stdout.flush() # Ensure registration message is flushed