# job_page_callbacks.py
import sys
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import pandas as pd
import traceback
from datetime import datetime, timedelta, timezone
from io import StringIO # Needed to read JSON from Store

from get_handler import get_handler # Assuming this helper exists

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

    # --- Add other callbacks for XAI or other plots if needed ---

    print("Job page callbacks registered.")
    sys.stdout.flush() # Ensure registration message is flushed