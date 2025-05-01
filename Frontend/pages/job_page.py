# pages/job_page.py
import dash
from dash import dcc, html
import plotly.graph_objects as go # Often needed for graph creation

# This is the main layout function called by app.py
def layout(handler, job_name):
    """
    Defines the layout for the individual job results page.
    All numeric columns are available in the graph legend but hidden by default.

    Args:
        handler: The FrontendHandler instance (passed from app.py).
        job_name (str): The name of the job being displayed.

    Returns:
        dash.html.Div: The layout components for the page.
    """
    print(f"Generating layout for job: {job_name}")

    # Note: The list of numeric columns is now determined dynamically
    # within the callback based on the fetched data. No need to define it here.

    return html.Div([
        # --- Store Components (keep as before) ---
        dcc.Store(id='job-page-job-name-store', data=job_name),
        dcc.Store(id='job-page-data-store'),
        dcc.Store(id='job-page-xai-store'), # Assuming you might use XAI later
        dcc.Store(id='job-page-status-store'), # Assuming you might use status later

        dcc.Interval(
            id='job-page-interval-component',
            interval=10*1000,  # Interval in milliseconds (e.g., 10 seconds)
            n_intervals=0,
            disabled=False # Keep enabled for potential streaming updates
        ),

        # --- Header and Navigation (keep as before) ---
        html.Div([
            html.H1(f"Analysis Results: {job_name}", style={'textAlign': 'center', 'color': '#E0E0E0'}),
            dcc.Link("<< Back to Home Page", href="/", style={'color': '#7FDBFF', 'fontSize': '18px'}),
            dcc.Loading(
                id="loading-job-page", type="circle", fullscreen=False,
                children=[html.Div(id="loading-output-jobpage")] # Target for loading indicator
            ),
        ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#1E3A5F', 'borderRadius': '5px'}),

        # --- Main Content Area ---
        html.Div([
            # Status Display (keep as before)
            html.Div(id='job-status-display', style={'marginBottom': '15px', 'padding': '10px', 'border': '1px solid #444', 'borderRadius': '5px', 'backgroundColor': '#145E88', 'color': '#FFFFFF'}),

            # --- DROPDOWN REMOVED ---
            # The html.Div containing the dcc.Dropdown with id='y-axis-dropdown' has been removed.

            # Graph for main timeseries data and anomalies
            html.Div([
                html.H3("Time Series Data & Detected Anomalies", style={'color': '#C0C0C0'}),
                # Added hint about legend interaction
                html.P("(Click legend items to toggle visibility)", style={'color': '#A0A0A0', 'fontSize':'small', 'textAlign':'center', 'marginTop': '-10px', 'marginBottom': '10px'}),
                dcc.Graph(id='timeseries-anomaly-graph', figure={}) # Initial empty figure
            ], style={'marginBottom': '20px'}),

            # XAI Section (keep as before, assuming placeholder for now)
            html.Div([
                html.H3("Explainability (XAI) Results", style={'color': '#C0C0C0'}),
                # Container where the callback will inject results
                html.Div(id='xai-results-content', children="Checking for XAI results...")
            ], id='xai-results-section', style={'display': 'block', 'marginBottom': '20px'}), # Keep visible initially or control via callback

            # Other plots placeholder (keep as before)
            html.Div(id='other-plots-section')

        ], style={'padding': '20px', 'backgroundColor': '#104E78', 'borderRadius': '10px'}),

    ], style={'padding': '30px', 'backgroundColor': '#0D3D66', 'minHeight': '100vh'})