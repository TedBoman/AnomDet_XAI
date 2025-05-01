# job_page_callbacks.py
import sys
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import pandas as pd
import traceback
from datetime import datetime, timedelta, timezone
from io import StringIO # Needed to read JSON from Store

from get_handler import get_handler

def register_job_page_callbacks(app):
    print("Registering job page callbacks...")

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
                # Split the path by '/' and take the last part
                job_name = pathname.split('/')[-1]
                # Basic validation: ensure it's not empty after split
                if job_name:
                    print(f"(URL Parser CB) Extracted job name: {job_name}. Updating store.")
                    return job_name # Update the store
                else:
                    print("(URL Parser CB) Job name empty after split.")
                    return None # Or return dash.no_update if you don't want to clear it
            except Exception as e:
                print(f"(URL Parser CB) Error parsing pathname '{pathname}': {e}")
                return None # Or return dash.no_update
        else:
            print("(URL Parser CB) Pathname doesn't match expected '/job/...' format.")
            # Decide what to do: clear the job name or leave it as is?
            # Clearing it might be safer if the user navigates away.
            return None # Return None if path doesn't match

    # --- Callback 1: Fetch and Store Data ---
    @app.callback(
        [
            Output('job-page-data-store', 'data'),
            Output('job-status-display', 'children'),
            Output('loading-output-jobpage', 'children')
        ],
        [
            Input('job-page-job-name-store', 'data'),       # Trigger on job change
            Input('job-page-interval-component', 'n_intervals') # Trigger on interval
        ]
        # No State needed
    )
    def update_data_store(job_name, n_intervals):
        """
        Fetches data based on triggers:
        - Always fetches when job_name changes.
        - Fetches periodically ONLY if job_name starts with 'job_stream_'.
        - Stores fetched data in dcc.Store. Updates status.
        """
        ctx = dash.callback_context
    
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if a valid job name is provided
        if not job_name:
            return None, f"No job selected. Last checked: {current_time_display}", None

        # Determine job type
        is_streaming_job = job_name.startswith("job_stream_")
        is_batch_job = job_name.startswith("job_batch_")
        triggered_by_job_change = trigger_id == 'job-page-job-name-store'
        triggered_by_interval = trigger_id == 'job-page-interval-component'

        # --- Decide whether to fetch ---
        should_fetch = False
        fetch_reason = ""
        start_time_iso = None # Initialize start time

        # NOTE: triggered_by_job_change will now be TRUE on initial load because
        # the update_job_store_from_url callback updates the store.

        if triggered_by_job_change:
            if is_streaming_job or is_batch_job:
                 should_fetch = True
                 fetch_reason = f"Job selected/changed to '{job_name}'"
                 # Determine start time for initial fetch on job change
                 if is_batch_job:
                      start_time_iso = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
                      print(f"(Data Fetch CB) Using epoch start for batch job '{job_name}'.")
                 else: # is_streaming_job (or fallback)
                      # Fetch recent data on initial load for streaming job (e.g., last hour)
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
                 # Fetch recent data for interval updates (e.g., last 5-10 minutes to catch up)
                 # Adjust lookback based on interval frequency to avoid gaps/overlap issues
                 # Or implement logic to fetch since last known timestamp (more complex)
                 lookback_minutes = 10 # Example: Look back 10 mins on interval
                 start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                 start_time_iso = start_time.isoformat()
                 print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming interval fetch.")
            elif is_batch_job:
                 # Interval triggered for batch job, DO NOTHING with data
                 print(f"(Data Fetch CB) Interval trigger ignored for batch job '{job_name}'.")
                 status_msg = f"Data loaded for batch job '{job_name}'. (No periodic update). Last checked: {current_time_display}"
                 # Return no_update to keep existing stored data and loading state, only update status
                 return dash.no_update, status_msg, dash.no_update
            else:
                 print(f"(Data Fetch CB) Interval trigger for unrecognized job type: {job_name}. No fetch.")

        # --- Perform Fetch if needed ---
        if should_fetch and start_time_iso:
            print(f"(Data Fetch CB) Condition met: {fetch_reason}. Fetching data from {start_time_iso}...")
            status_msg = f"Fetching data for job '{job_name}' ({'Streaming' if is_streaming_job else 'Batch'})..."
            data_json = None
            try:
                handler = get_handler()
                df = handler.handle_get_data(timestamp=start_time_iso, job_name=job_name)

                if not df.empty:
                    print(f"(Data Fetch CB) Successfully fetched data. Shape: {df.shape}")
                    data_json = df.to_json(date_format='iso', orient='split')
                    status_msg = f"Data updated for job '{job_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}"
                else:
                    print(f"(Data Fetch CB) Received empty DataFrame for job '{job_name}'.")
                    status_msg = f"No new data found for job '{job_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}"
                    # Keep old data or clear? If streaming, maybe keep old + append new?
                    # Current simple approach: Return None if empty df received. This will clear graph.
                    # A more robust approach would merge new data with data already in the store.
                    data_json = None # Set to None to clear graph if no data received

            except Exception as e:
                print(f"(Data Fetch CB) Error fetching data for job '{job_name}':")
                traceback.print_exc()
                status_msg = f"Error fetching data for job '{job_name}': {e}. Timestamp: {current_time_display}"
                data_json = None # Clear data on error

            return data_json, status_msg, None # Return data (or None), status, clear loading

        else:
            # No fetch condition met or start_time not set, return no_update
            print("(Data Fetch CB) No fetch condition met or start_time_iso not set. Returning no_update.")
            return dash.no_update, dash.no_update, dash.no_update


    # --- Callback 2: Update Graph and Dropdown Options from Stored Data ---
    @app.callback(
        [
            Output('timeseries-anomaly-graph', 'figure'),
            Output('y-axis-dropdown', 'options')
        ],
        [
            Input('job-page-data-store', 'data'), # Trigger when stored data changes
            Input('y-axis-dropdown', 'value')     # Trigger when dropdown selection changes
        ],
        [State('job-page-job-name-store', 'data')] # Get job name for title
    )
    def update_graph_and_options(stored_data_json, selected_y_columns, job_name):
        """
        Updates the graph and dropdown options based on the data stored
        in dcc.Store and the user's dropdown selection.
        """
        # Default empty figure and options
        fig = go.Figure(layout={'template': 'plotly_dark', 'title': 'Select Job and Columns'})
        dropdown_options = []
        job_title_name = job_name if job_name else "No Job"

        if stored_data_json is None:
            print("(Graph Update CB) No data in store.")
            fig.update_layout(title=f'No Data Available for {job_title_name}', xaxis={'visible': False}, yaxis={'visible': False})
            return fig, [] # Return empty figure and options

        print("(Graph Update CB) Data found in store, processing graph and options...")
        try:
            # --- Read DataFrame from stored JSON ---
            df = pd.read_json(StringIO(stored_data_json), orient='split')
            print(f"(Graph Update CB) DataFrame loaded. Shape: {df.shape}")

            # --- IMPORTANT: Verify Timestamp Column ---
            if 'timestamp' not in df.columns:
                raise ValueError("'timestamp' column not found in DataFrame")
            
            # Define the known start time (epoch) - make it timezone-aware (UTC)
            # Use timezone.utc for clarity
            epoch_start = pd.Timestamp('1970-01-01 00:00:00', tzinfo=timezone.utc)

            # Convert the numeric 'timestamp' column (ASSUMING it's seconds) to timedelta and add to epoch start
            # Use pd.to_numeric to handle potential strings first, coercing errors
            relative_seconds = pd.to_numeric(df['timestamp'], errors='coerce')
            if relative_seconds.isnull().any():
                print("Warning: Some timestamp values were non-numeric and converted to NaT.")

            # Add the timedelta (in seconds) to the epoch start time
            df['timestamp'] = epoch_start + pd.to_timedelta(relative_seconds, unit='s')

            print(f"(Graph Update CB) Converted timestamp head:\n{df['timestamp'].head()}")
            print(f"(Graph Update CB) Converted timestamp dtype: {df['timestamp'].dtype}") # Should be datetime64[ns, UTC]

            # --- Populate Dropdown Options ---
            cols_to_exclude = {'timestamp', 'label', 'is_anomaly', 'injected_anomaly'}
            numeric_cols = df.select_dtypes(include=['number']).columns # Step 1: Find numeric cols
            available_y_cols = [col for col in numeric_cols if col not in cols_to_exclude] # Step 2: Filter excluded
            dropdown_options = [{'label': col, 'value': col} for col in available_y_cols] # Step 3: Create options list
            print(f"(Graph Update CB) Generated dropdown options: {len(dropdown_options)} columns.")

            # --- Create Graph ---
            fig = go.Figure(layout=go.Layout(
                title=f"Time Series Data for {job_title_name}",
                template="plotly_dark",
                xaxis_title="Timestamp",
                yaxis_title="Selected Values",
                uirevision=job_name # Preserve zoom/pan for the same job
            ))

            x_axis_data = df['timestamp']

            if selected_y_columns: # Check if user selected any columns
                print(f"(Graph Update CB) Plotting selected columns: {selected_y_columns}")
                for col_name in selected_y_columns:
                    if col_name in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            fig.add_trace(go.Scattergl(
                                x=x_axis_data,
                                y=df[col_name],
                                mode='lines', # Use 'lines+markers' if you want points too
                                name=col_name
                            ))
                            print(f"(Graph Update CB) SUCCESSFULLY added trace for: {col_name}")
                        else:
                            print(f"Warning: Selected column '{col_name}' is not numeric. Skipping.")
                    else:
                        print(f"Warning: Selected column '{col_name}' not found in DataFrame. Skipping.")
            else:
                print("(Graph Update CB) No Y-axis columns selected.")
                fig.update_layout(
                    title=f"Data Loaded for {job_title_name} - Select Columns to Plot",
                    xaxis={'visible': True, 'title': 'Timestamp (Select Y-axis)'}, # Show timestamp axis maybe?
                    yaxis={'visible': False},
                    annotations=[dict(
                        text="Select columns from the dropdown above.",
                        xref="paper", yref="paper",
                        showarrow=False, font=dict(size=16)
                    )]
                )

        except Exception as e:
            print(f"(Graph Update CB) Error processing stored data or plotting:")
            traceback.print_exc()
            fig = go.Figure(layout={
                'template': 'plotly_dark', 'title': f'Error Displaying Data for {job_title_name}',
                'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                'annotations': [{'text': f'An error occurred: {e}', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
            })
            # Keep existing options or clear them? Let's clear on error.
            dropdown_options = []
            sys.stdout.flush()

        return fig, dropdown_options

    sys.stdout.flush()
    print("Job page callbacks registered.")