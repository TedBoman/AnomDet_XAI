# pages/job_page.py
import dash
from dash import dcc, html, Input, Output, State

# This is the main layout function called by app.py
def layout(handler, job_name):
    """
    Defines the layout for the individual job results page.
    Includes a dropdown for selecting columns to plot.

    Args:
        handler: The FrontendHandler instance (passed from app.py).
        job_name (str): The name of the job being displayed.

    Returns:
        dash.html.Div: The layout components for the page.
    """
    print(f"Generating layout for job: {job_name}")

    # Define known numeric columns from the creditcard dataset for the dropdown
    # Excludes 'Time', 'Class'/'label', 'is_anomaly', 'injected_anomaly'
    numeric_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    dropdown_options = [{'label': col, 'value': col} for col in numeric_columns]

    return html.Div([
        # --- Store Components (keep as before) ---
        dcc.Store(id='job-page-job-name-store', data=job_name),
        dcc.Store(id='job-page-data-store'),
        dcc.Store(id='job-page-xai-store'),
        dcc.Store(id='job-page-status-store'),

        dcc.Interval(
        id='job-page-interval-component',
        interval=10*1000,  # Interval in milliseconds (e.g., 10 seconds = 10 * 1000ms)
        n_intervals=0,     # Initial value, doesn't matter much
        disabled=False     # Keep it enabled so it ticks for streaming jobs
                           # The callback logic already ignores ticks for batch jobs
        ),

        # --- Header and Navigation (keep as before) ---
        html.Div([
            html.H1(f"Analysis Results: {job_name}", style={'textAlign': 'center', 'color': '#E0E0E0'}),
            dcc.Link("<< Back to Home Page", href="/", style={'color': '#7FDBFF', 'fontSize': '18px'}),
            dcc.Loading(
                id="loading-job-page", type="circle", fullscreen=False,
                children=[html.Div(id="loading-output-jobpage")]
            ),
        ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#1E3A5F', 'borderRadius': '5px'}),

        # --- Main Content Area ---
        html.Div([
            # Status Display (keep as before)
            html.Div(id='job-status-display', style={'marginBottom': '15px', 'padding': '10px', 'border': '1px solid #444', 'borderRadius': '5px', 'backgroundColor': '#145E88', 'color': '#FFFFFF'}),

            # --- !!! ADDED COLUMN SELECTOR DROPDOWN !!! ---
            html.Div([
                html.Label("Select Columns to Plot:", style={'marginRight': '10px', 'color': '#E0E0E0', 'fontWeight':'bold'}),
                dcc.Dropdown(
                    id='y-axis-dropdown',
                    options=dropdown_options,
                    value=['Amount'],  # Default value (e.g., plot 'Amount' initially)
                    multi=True,       # Allow multiple selections
                    style={'width': '80%', 'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#333'}
                )
            ], style={'marginBottom': '20px', 'textAlign': 'center'}),
            # --- !!! END ADDED DROPDOWN !!! ---

            # Graph for main timeseries data and anomalies
            html.Div([
                html.H3("Time Series Data & Detected Anomalies", style={'color': '#C0C0C0'}),
                dcc.Graph(id='timeseries-anomaly-graph', figure={})
            ], style={'marginBottom': '20px'}),

            # XAI Section (keep as before)
            html.Div([
                 html.H3("Explainability (XAI) Results", style={'color': '#C0C0C0'}),
                 dcc.Graph(id='xai-feature-importance-graph', figure={}),
                 html.Div(id='xai-other-results-display')
             ], id='xai-results-section', style={'display': 'none', 'marginBottom': '20px'}),

            # Other plots placeholder (keep as before)
            html.Div(id='other-plots-section')

        ], style={'padding': '20px', 'backgroundColor': '#104E78', 'borderRadius': '10px'}),

    ], style={'padding': '30px', 'backgroundColor': '#0D3D66', 'minHeight': '100vh'})