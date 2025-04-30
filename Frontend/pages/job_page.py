# pages/job_page.py
import dash
from dash import dcc, html, Input, Output, State
# Might need Plotly Express or Graph Objects later for creating figures
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd # Likely needed in callbacks

# This is the main layout function called by app.py
def layout(handler, job_name):
    """
    Defines the layout for the individual job results page.

    Args:
        handler: The FrontendHandler instance (passed from app.py).
        job_name (str): The name of the job being displayed.

    Returns:
        dash.html.Div: The layout components for the page.
    """
    print(f"Generating layout for job: {job_name}") # Debug print

    # --- Fetch initial static info about the job if needed ---
    # Example: Get job type (batch/stream) or parameters
    # Note: It's generally better to fetch dynamic data via callbacks
    # job_details = handler.handle_get_job_details(job_name) # Assuming such a handler method exists
    # job_type = job_details.get("type", "Unknown")

    return html.Div([
        # --- Store Components ---
        dcc.Store(id='job-page-job-name-store', data=job_name),
        dcc.Store(id='job-page-data-store'), # Store for main data
        dcc.Store(id='job-page-xai-store'), # Store for XAI results (if added later)
        dcc.Store(id='job-page-status-store'), # Store for job status

        # --- Header and Navigation ---
        html.Div([
            html.H1(f"Analysis Results: {job_name}", style={'textAlign': 'center', 'color': '#E0E0E0'}),
            dcc.Link("<< Back to Home Page", href="/", style={'color': '#7FDBFF', 'fontSize': '18px'}),
             # Add a loading spinner overlay for callbacks
            dcc.Loading(
                id="loading-job-page",
                type="circle",
                fullscreen=False,
                children=[html.Div(id="loading-output-jobpage")] # Dummy output
            ),
        ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#1E3A5F', 'borderRadius': '5px'}),

        html.Div([
            # Placeholder for displaying job status or key metrics
            html.Div(id='job-status-display', style={'marginBottom': '15px', 'padding': '10px', 'border': '1px solid #444', 'borderRadius': '5px', 'backgroundColor': '#145E88', 'color': '#FFFFFF'}),

            # Graph for main timeseries data and anomalies
            html.Div([
                html.H3("Time Series Data & Detected Anomalies", style={'color': '#C0C0C0'}),
                # Graph component where the Plotly figure will be rendered
                dcc.Graph(id='timeseries-anomaly-graph', figure={}) # Initialize with empty figure
            ], style={'marginBottom': '20px'}),


            # Section for XAI results (initially hidden, can be populated later)
            html.Div([
                html.H3("Explainability (XAI) Results", style={'color': '#C0C0C0'}),
                dcc.Graph(id='xai-feature-importance-graph', figure={}), # Example XAI graph
                html.Div(id='xai-other-results-display') # For text or other outputs
            ], id='xai-results-section', style={'display': 'none', 'marginBottom': '20px'}), # Start hidden

            # Placeholder for other plots or information
            html.Div(id='other-plots-section')

        ], style={'padding': '20px', 'backgroundColor': '#104E78', 'borderRadius': '10px'}),

        dcc.Interval(
            id='job-page-interval-component',
            interval=5*1000,  # Update every 5 seconds (adjust as needed)
            n_intervals=0,
            disabled=False # Enable by default
        ),

    ], style={'padding': '30px', 'backgroundColor': '#0D3D66', 'minHeight': '100vh'}) # Basic page styling