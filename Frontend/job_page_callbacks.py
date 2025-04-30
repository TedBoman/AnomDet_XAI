# job_page_callbacks.py 
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go
# import plotly.express as px # Uncomment if using Plotly Express
# from plotly.subplots import make_subplots # Uncomment if needed
import pandas as pd
import json
from io import StringIO
import traceback # For error printing
from datetime import datetime # For status message timestamp

# Import your handler getter
from get_handler import get_handler

def register_job_page_callbacks(app):
    print("Registering job page callbacks...")

    @app.callback(
        [
            Output('timeseries-anomaly-graph', 'figure'),
            Output('job-page-data-store', 'data'), # Still need to provide output
            Output('loading-output-jobpage', 'children'),
            Output('job-status-display', 'children'),
        ],
        [Input('job-page-interval-component', 'n_intervals')],
        [State('job-page-job-name-store', 'data')]
    )
    def update_job_data_and_graphs(n_intervals, job_name):
        """
        TEMPORARY callback to test dcc.Graph rendering with a minimal figure.
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

        print(f"Minimal figure test callback triggered for job: {job_name}, interval: {n_intervals}, trigger: {trigger_id}")

        if not job_name:
            # Return empty figure if no job name
            return go.Figure(layout={'template': 'plotly_dark', 'title': 'No Job Selected'}), no_update, None, "No job selected."

        # --- !!! CREATE HARDCODED FIGURE !!! ---
        print("--- Creating minimal hardcoded figure ---")
        minimal_fig = go.Figure(
            data=[go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines+markers', name='Test Trace')],
            layout=go.Layout(
                title=f"Minimal Test Plot for {job_name}",
                template="plotly_dark",
                xaxis_title="Test X",
                yaxis_title="Test Y"
            )
        )
        # --- !!! END HARDCODED FIGURE !!! ---

        # Update status message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_msg = f"Minimal test update for job '{job_name}'. Last checked: {current_time}"

        # Return the minimal figure and dummy outputs for other Outpupts
        # Return None for the data store as we didn't fetch data
        return minimal_fig, None, None, status_msg

    # --- Add other callbacks for XAI, etc. here later if needed ---
    # @app.callback(...)
    # def update_xai_display(...):
    #     ...

    print("Job page callbacks registered.")