import os
import requests
import socket
import json
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
from dash.dependencies import ALL
from callbacks import create_active_jobs

def layout(handler):
    try:
        datasets = handler.handle_get_datasets()
    except Exception as e:
        print(f"Error getting datasets: {e}")
        datasets = []
    try:
        models = handler.handle_get_models()
    except Exception as e:
        print(f"Error getting models: {e}")
        models = []
    try:
        xai_methods = handler.handle_get_xai_methods()
    except Exception as e:
        print(f"Error getting XAI methods: {e}")
        xai_methods = []
    try:
        injection_methods = handler.handle_get_injection_methods()
    except Exception as e:
        print(f"Error getting injection methods: {e}")
        injection_methods = []
    try:
        active_jobs_data = handler.handle_get_running()
        active_jobs = json.loads(active_jobs_data)["running"] if active_jobs_data else []
    except Exception as e:
        print(f"Error getting running jobs: {e}")
        active_jobs = []

    active_jobs_children = create_active_jobs(active_jobs)
    got_jobs = len(active_jobs) > 0

    layout = html.Div([

        # --- Hyperparameter Explanation Box ---
        html.Div(
            id="parameter-explanation-box",
            children=[html.P("Select a model to see parameter explanations.", style={'color':'#b0b0b0'})],
            style={
                "padding": "20px",
                "backgroundColor": "#0a3d5a", # Slightly different background
                "borderRadius": "1rem",
                "boxShadow": "0 2px 5px rgb(0, 0, 0)",
                "minWidth": "300px", # Adjust width as needed
                "maxWidth": "400px",
                "height": "fit-content", # Adjust height based on content
                "maxHeight": "80vh", # Limit max height and make scrollable if needed
                "overflowY": "auto", # Add scroll if content exceeds maxHeight
                # Add flex properties if needed, e.g., flex: 1 (takes up 1/3 width)
                "color": "#e0e0e0",
                "textAlign": "left",
            }
        ),
        # --- END Explanation Box ---

        # --- AnomDetX Settings panel ---
        html.Div( 
            [ 
                html.H1("AnomDetX", style={
                    "textAlign": "center",
                    "marginBottom": "30px",
                    "color": "#ffffff",
                    "fontSize": "3.5rem"
                }),

                # --- NEW: Parent Div for settings and explanations ---
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label("Select Dataset:", style={"fontSize": "22px", "color": "#ffffff"}),
                            dcc.Dropdown(
                                id="dataset-dropdown",
                                options=[{"label": dataset, "value": dataset} for dataset in datasets],
                                value=None,
                                placeholder="Select a dataset",
                                style={"width": "350px", "fontSize": "18px", "margin": "auto", "border": "0.05rem solid black"}
                            )
                        ], style={"textAlign": "center", "marginBottom": "30px"}),

                        # --- Labeled Dataset Section ---
                        html.Div([
                            dcc.Checklist(
                                id="labeled-check",
                                options=[{"label": "Is dataset labeled for anomalies?", "value": "is_labeled"}],
                                value=[], # Initially unchecked
                                style={"fontSize": "20px", "color": "#ffffff", "marginBottom": "10px"}
                            ),
                            # Div to hold the conditional label column dropdown
                            html.Div(
                                id="label-column-selection-div",
                                children=[
                                    html.Label("Select Label Column:", style={"fontSize": "18px", "color": "#e0e0e0", "display": "block"}),
                                    dcc.Dropdown(
                                        id="label-column-dropdown", # NEW ID for clarity
                                        options=[], # Populated by callback
                                        value=None, # Reset by callback
                                        placeholder="Select label column",
                                        multi=False, # Select only ONE label column
                                        style={"width": "300px", "margin": "5px auto", "border": "0.05rem solid black"}
                                    )
                                ],
                                # Initially hidden, shown by callback
                                style={"display": "none", "marginTop": "10px", "textAlign": "center"}
                            )
                        ], style={"textAlign": "center", "marginBottom": "20px", "padding": "10px", "border": "1px dashed #555", "borderRadius": "5px"}),

                        # --- Detection Model Selection ---
                        html.Div([
                            html.Label("Select a Detection Model:", style={"fontSize": "22px", "color": "#ffffff"}),
                            dcc.Dropdown(
                                id="detection-model-dropdown",
                                options=[{"label": model, "value": model} for model in models],
                                placeholder="Select a detection model",
                                style={"width": "350px", "fontSize": "18px", "margin": "auto", "border": "0.05rem solid black"}
                            )
                        ], style={"textAlign": "center", "marginTop": "30px"}),

                        # --- Placeholder for Model Settings ---
                        html.Div(
                            id="model-settings-panel",
                            children=[], # Populated by callback
                            style={"marginTop": "15px", "padding": "15px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88", "display": "none"} # Start hidden
                        ),

                        # --- XAI Section ---
                        html.Div([
                            dcc.Checklist(
                                id="xai-check",
                                options=[{"label": "Run Explainability (XAI)?", "value": "use_xai"}], # Changed value for clarity
                                value=[], # Initially unchecked
                                style={"fontSize": "20px", "color": "#ffffff", "marginBottom": "10px"}
                            ),
                            # Div to hold conditional XAI options
                            html.Div(
                                id="xai-options-div",
                                children=[
                                    # XAI Method Selection
                                    html.Div([
                                        html.Label("Select XAI Method:", style={"fontSize": "18px", "color": "#e0e0e0", "display": "block"}),
                                        dcc.Dropdown(
                                            id="xai-method-dropdown",
                                            options=[{"label": method, "value": method} for method in xai_methods],
                                            value="none", # Default to None
                                            placeholder="Select XAI method",
                                            multi=True,
                                            clearable=False,
                                            style={"width": "300px", "margin": "5px auto"}
                                        ),
                                    ], style={"marginTop": "10px"}),
                                    # Div for Dynamic XAI Settings
                                    html.Div(
                                        id="xai-settings-panel",
                                        children=[], # Populated by callback
                                        style={"marginTop": "15px", "padding": "10px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88"}
                                    )
                                ],
                                # Initially hidden, shown by callback
                                style={"display": "none", "marginTop": "10px", "textAlign": "center"}
                            )
                        ], style={"textAlign": "center", "marginBottom": "20px", "padding": "10px", "border": "1px dashed #555", "borderRadius": "5px"}),

                        # --- Injection Section ---
                        html.Div([
                            dcc.Checklist( id="injection-check", options=[{"label": "Inject Anomalies?", "value": "use_injection"}], value=[],
                                        style={"fontSize": "20px", "color": "#ffffff", "marginBottom": "10px"} ),
                            html.Div( id="injection-panel", children=[
                                html.Label("Select Injection Method:", style={"fontSize": "18px", "color": "#e0e0e0", "display":"block"}),
                                dcc.Dropdown( id="injection-method-dropdown", options=[{"label": method, "value": method} for method in injection_methods], value="None", placeholder="Select a method",
                                            style={"width": "300px", "margin": "5px auto"} ),
                                
                                html.Div([
                                    html.Label("Select Columns from Dataset:", style={"fontSize": "18px", "color": "#ffffff"}),
                                    dcc.Dropdown(
                                        id="injection-column-dropdown",
                                        options=[],
                                        value=[],
                                        placeholder="Select Columns",
                                        multi=True,
                                        style={"width": "300px", "margin": "5px auto"}
                                    )
                                ], style={"marginTop": "15px"}),
                                html.Div([
                            html.Label("Select Time for Timestamp (seconds since epoch):", style={"fontSize": "18px", "color": "#ffffff"}),
                            dcc.Input(
                                id="timestamp-input",
                                type="number",
                                placeholder="seconds since epoch",
                                style={"width": "200px", "marginTop": "10px"}
                            ) 
                            ], style={"marginTop": "20px", "textAlign": "center"}),
                            html.Div([
                                html.Label("Enter Magnitude (Default: 1):", style={"fontSize": "18px", "color": "#ffffff"}),
                                dcc.Input(
                                    id="magnitude-input",
                                    type="number",
                                    placeholder="Magnitude",
                                    value=1,
                                    style={"width": "200px", "marginTop": "10px"}
                                )
                            ], style={"marginTop": "20px", "textAlign": "center"}),
                            html.Div([
                                html.Label("Enter Anomaly Percentage (%):", style={"fontSize": "18px", "color": "#ffffff"}),
                                dcc.Input(
                                    id="percentage-input",
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=1,
                                    placeholder="Percentage",
                                    style={"width": "200px", "marginTop": "10px"}
                                )
                            ], style={"marginTop": "20px", "textAlign": "center"}),
                            html.Div([
                                html.Label("Enter a duration: ", style={"fontSize": "18px", "color": "#ffffff"}),
                                dcc.Input(
                                    id="duration-input",
                                    type="text",
                                    placeholder="Duration ('30s', '1H', '30min', '2D', '1h30m')",
                                    style={"width": "200px", "marginTop": "10px"}
                                )
                                ], style={"marginTop": "20px", "textAlign": "center"})
                            ], style={"display": "none"} # injection-panel starts hidden
                            )
                        ], style={"textAlign": "center", "marginBottom": "20px", "padding": "10px", "border": "1px dashed #555", "borderRadius": "5px"}),
                            
                        # --- Job Naming ---
                        html.Div([
                            html.Label("Job name: ", style={"fontSize": "22px", "color": "#ffffff"}),
                            dcc.Input(
                                        id="name-input",
                                        type="text",
                                        placeholder="JOB_NAME",
                                        value="",
                                        style={"width": "200px", "marginTop": "10px"}
                                    )
                        ], style={"display": "block", "marginTop": "15px", "textAlign": "center"}),

                        # --- Mode Selection (Batch/Stream) ---
                        html.Div([
                            html.Label("", style={}),
                            dcc.RadioItems(
                                id="mode-selection",
                                options=[
                                    {"label": "Batch", "value": "batch"},
                                    {"label": "Stream", "value": "stream"}
                                ],
                                value="batch",  # Default selection
                                labelStyle={"display": "inline-block", "marginRight": "20px"},
                                style={"textAlign": "center", "fontSize": "25px", "color": "#ffffff"},
                                inputStyle={"height": "22px", "width": "30px", "marginRight": "10px"}
                            )
                        ], style={"textAlign": "center", "marginTop": "20px"}),

                        # --- Speedup Input (Conditional) ---
                        html.Div([
                            html.Label("Select Speedup for Stream (Default: 1):", style={"fontSize": "22px", "color": "#ffffff"}),
                            dcc.Input(
                                id="speedup-input",
                                type="number",
                                value=1,
                                step=0.1,
                                style={"width": "200px", "marginTop": "10px"}
                            )
                        ], style={"marginTop": "20px", "textAlign": "center"}),
                        
                        # --- Start Button ---
                        html.Div([
                            html.Button("Start Job", id="start-job-btn", style={
                                "marginTop": "20px",
                                "width": "150px",
                                "height": "40px",
                                "fontSize": "16px",
                                "backgroundColor": "#4CAF50",
                                "color": "#ffffff",
                                "borderRadius": "0.3rem",
                                "display": "block",
                                "margin": "auto"
                            })
                        ], style={"textAlign": "center", "marginTop": "30px"}), 

                        # --- Popups and Intervals ---
                        html.Div(
                            id="popup",
                            children="Job has started!",
                            style={
                                "backgroundColor": "#4CAF50",
                                "color": "#ffffff",
                                "fontSize": "20px",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "width": "250px",
                                "margin": "auto",
                                "position": "fixed",
                                "top": "20px",
                                "left": "50%",
                                "transform": "translateX(-50%)",
                                "zIndex": "1000",
                                "display": "none"
                            }),
                            dcc.Interval(
                            id="popup-interval",
                            interval=3 * 1000,
                            n_intervals=0,
                            disabled=True 
                            ),
                        ]),
                # --- End Main settings panel

        ], style={ # Style for the parent flex container
             "display": "flex",
             "flexDirection": "row", # Arrange items side-by-side
             "justifyContent": "center", # Center the items horizontally
             "alignItems": "flex-start", # Align items to the top
             "maxWidth": "80rem", # Adjust overall max width if needed
             "margin": "auto", # Center the whole container
        }),


            # --- Active Jobs ---
            html.Div(
                id="active-jobs-section",
                children=[
                    html.H3("Currently Running Jobs:", style={"color": "#ffffff", "textAlign": "center"}),
                    html.Div(children=active_jobs_children, id="active-jobs-list", style={
                        "textAlign": "center", "color": "#ffffff", "marginTop": "4px",
                        "width": "25rem", "margin": "10px auto", "padding": "10px", "border": "4px solid #464", "borderRadius": "5px"
                    }),
                    dcc.Store(id='active-jobs-json', data=""),
                    dcc.Interval(
                        id="job-interval",
                        interval=5 * 1000,
                        n_intervals=0,
                        disabled=False 
                    )
                ],
                style={"display": "block", "marginTop": "30px"} if got_jobs else {"display": "none"} # Hidden by default
            ),
        ],style={
            "padding": "30px",
            "backgroundColor": "#104E78",
            "maxWidth": "40rem",
            "borderRadius": "2rem",
            "margin": "auto",
            "boxShadow": "0 4px 10px rgb(0, 0, 0)",
            "textAlign": "center",
        }),

        # --- XAI method Explanation Box ---
        html.Div(
            id="xai-explanation-box",
            children=[html.P("Select a XAI method to see parameter explanations.", style={'color':'#b0b0b0'})],
            style={
                "padding": "20px",
                "backgroundColor": "#0a3d5a", # Slightly different background
                "borderRadius": "1rem",
                "boxShadow": "0 2px 5px rgb(0, 0, 0)",
                "minWidth": "300px", # Adjust width as needed
                "maxWidth": "400px",
                "height": "fit-content", # Adjust height based on content
                "maxHeight": "80vh", # Limit max height and make scrollable if needed
                "overflowY": "auto", # Add scroll if content exceeds maxHeight
                # Add flex properties if needed, e.g., flex: 1 (takes up 1/3 width)
                "color": "#e0e0e0",
                "textAlign": "left",
            }
        ),
        # --- Explanation Box ---

    ], id="main-settings-container", style={ "backgroundColor": "#105E90", "padding": "40px", "minHeight": "100vh", "display": "flex", "flexWrap": "nowrap", "alignItems": "center"})

    return layout