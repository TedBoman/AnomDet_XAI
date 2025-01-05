import os
import requests
import socket
import json
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
from dash.dependencies import ALL

def layout(handler):
    datasets = handler.handle_get_datasets()
    models = handler.handle_get_models()
    injection_methods = handler.handle_get_injection_methods()

    layout = html.Div([
        html.Div(
            [
                html.H1("AnomDet", style={
                    "textAlign": "center",
                    "marginBottom": "30px",
                    "color": "#ffffff",
                    "fontSize": "3.5rem"
                }),

                html.Div([
                    html.Label("Select Dataset:", style={"fontSize": "22px", "color": "#ffffff"}),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        options=[{"label": dataset, "value": dataset} for dataset in datasets],
                        placeholder="Select a dataset",
                        style={"width": "350px", "fontSize": "18px", "margin": "auto", "border": "0.05rem solid black"}
                    )
                ], style={"textAlign": "center", "marginBottom": "30px"}),

                html.Div([
                    html.Label("Select a Detection Model:", style={"fontSize": "22px", "color": "#ffffff"}),
                    dcc.Dropdown(
                        id="detection-model-dropdown",
                        options=[{"label": model, "value": model} for model in models],
                        placeholder="Select a detection model",
                        style={"width": "350px", "margin": "auto", "border": "0.05rem solid black"}
                    )
                ], style={"textAlign": "center", "marginTop": "30px"}),

                # Injection Checkbox
                html.Div([
                    dcc.Checklist(
                        id="injection-check",
                        options=[{"label": "Use Injection", "value": "use_injection"}],
                        value=[],
                        style={"textAlign": "center", "fontSize": "20px", "color": "#ffffff"}
                    ),
                    html.Div(id="injection-panel", style={"display": "none"})
                ], style={"marginTop": "30px"}),


                html.Div(children=[
                    html.Label("Select an Injection Method:", style={"fontSize": "22px", "color": "#ffffff"}),
                    dcc.Dropdown(
                        id="injection-method-dropdown",
                        options=[{"label": method, "value": method} for method in injection_methods],
                        value="None",
                        placeholder="Select a method",
                        style={"width": "350px", "margin": "auto"}
                    ),
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
                            type="number",
                            placeholder="Duration ('30s', '1H', '30min', '2D', '1h30m')",
                            style={"width": "200px", "marginTop": "10px"}
                        )
                        ], style={"marginTop": "20px", "textAlign": "center"}),
                    html.Div([
                        html.Label("Select Columns from Dataset:", style={"fontSize": "18px", "color": "#ffffff"}),
                        dcc.Dropdown(
                            id="column-dropdown",
                            options=[],
                            value=[],
                            placeholder="Select Columns",
                            multi=True,
                            style={"width": "350px", "marginTop": "10px"}
                        )
                        ], style={"marginTop": "20px", "textAlign": "center"}),
                    ], id="injection", style={"display": "none"}),

                html.Div([
                    html.Label("Job name: ", style={"fontSize": "22px", "color": "#ffffff"}),
                    dcc.Input(
                                id="name-input",
                                type="text",
                                placeholder="JOB_NAME",
                                style={"width": "200px", "marginTop": "10px"}
                            )
                ], style={"display": "block", "marginTop": "15px", "textAlign": "center"}),

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
                    interval=3000,
                    n_intervals=0,
                    disabled=True 
                    ),
                
                


                # Active Datasets Section
            html.Div(
                id="active-jobs-section",
                children=[
                    html.H3("Currently Running Jobs:", style={"color": "#ffffff", "textAlign": "center"}),
                    html.Div(id="active-jobs-list", style={
                        "textAlign": "center", "color": "#ffffff", "marginTop": "4px",
                        "width": "25rem", "margin": "10px auto", "padding": "10px", "border": "4px solid #464", "borderRadius": "5px"
                    }),
                    dcc.Interval(
                        id="job-interval",
                        interval=2000,
                        n_intervals=0,
                        disabled=False 
                    )
                ],
                style={"display": "none", "marginTop": "30px"}  # Hidden by default
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

    ], style={ "backgroundColor": "#105E90", "padding": "40px", "minHeight": "100vh"})

    return layout