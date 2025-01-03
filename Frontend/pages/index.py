import os
import requests
import socket
from frontend_handler import FrontendHandler
import json
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
from dash.dependencies import ALL
from dotenv import load_dotenv

load_dotenv()
HOST = 'Backend'
PORT = int(os.getenv('BACKEND_PORT'))

def layout():
    handler = FrontendHandler(HOST, PORT)

    datasets = handler.handle_get_datasets()
    models = handler.handle_get_models()
    injection_methods = handler.handle_get_injection_methods()
    active_datasets = handler.handle_get_running()

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
                                    placeholder="Select a method",
                                    style={"width": "350px", "margin": "auto"}
                                ),
                                html.Div([
                                    html.Label("Select Date Range:", style={"fontSize": "18px", "color": "#ffffff"}),
                                    dcc.DatePickerRange(
                                        id="date-picker-range",
                                        start_date_placeholder_text="Start Date",
                                        end_date_placeholder_text="End Date",
                                        display_format="YYYY-MM-DD",
                                        style={"marginTop": "10px"}
                                    )
                                ], style={"marginTop": "20px", "textAlign": "center"}),
                                html.Div([
                                html.Label("Select Time for Timestamp (Hour):", style={"fontSize": "18px", "color": "#ffffff"}),
                                dcc.Input(
                                    id="timestamp-hour",
                                    type="number",
                                    min=0,
                                    max=23,
                                    step=1,
                                    placeholder="Hour (0-23)",
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
                                    html.Label("Select Column from Dataset:", style={"fontSize": "18px", "color": "#ffffff"}),
                                    dcc.Dropdown(
                                        id="column-dropdown",
                                        options=[
                                            {"label": "Column 1", "value": "column_1"},
                                            {"label": "Column 2", "value": "column_2"}
                                        ],
                                        placeholder="Select a Column",
                                        style={"width": "350px", "marginTop": "10px"}
                                    )
                                ], style={"marginTop": "20px", "textAlign": "center"}),
                                html.Div([
                                    html.Label("Enter Duration (in seconds):", style={"fontSize": "18px", "color": "#ffffff"}),
                                    dcc.Input(
                                        id="duration-input",
                                        type="number",
                                        placeholder="Duration (sec)",
                                        style={"width": "200px", "marginTop": "10px"}
                                    )
                                ], style={"marginTop": "20px", "textAlign": "center"}),
                            ], id="injection", style={"display": "none"}),


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
                            html.Button("Start Job", id="add-dataset-btn", style={
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
                            html.Div(id="active-datasets-list", style={
                                "textAlign": "center", "color": "#ffffff", "marginTop": "4px",
                                "width": "25rem", "margin": "10px auto", "padding": "10px", "border": "4px solid #464", "borderRadius": "5px"
                            })
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

            ], style={
                "backgroundColor": "#105E90",#5187a8
                "padding": "40px",
                "minHeight": "100vh",

            })
    return layout

@callback(
    Output("injection", "style"),
    Input("injection-check", "value")
)
def update_injection_panel(selected):
    if "use_injection" in selected:
        return {"display": "block"}
    return {"display": "none"}


@callback(
    Output("active-jobs-section", "style"),
    Input("active-datasets-list", "children")
)
def toggle_active_jobs_section(children):
    # Visa sektionen om det finns några aktiva jobb, annars dölj den
    if children:
        return {"display": "block", "marginTop": "30px"}
    return {"display": "none"}

# Callback to add and manage active datasets
@callback(
    Output("active-datasets-list", "children"),
    [Input("add-dataset-btn", "n_clicks"),
     Input({"type": "remove-dataset-btn", "index": ALL}, "n_clicks")],
    [State("dataset-dropdown", "value")]
)
def manage_active_datasets(add_clicks, remove_clicks, selected_dataset):
    global active_datasets
    ctx = dash.callback_context

    if not ctx.triggered:
        return [
            html.Div([
                dcc.Link(
                    dataset,
                    href=f"/stream-data",
                    style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
                ),
                html.Button("Stop", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                    "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                    "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
                })
            ]) for dataset in active_datasets
        ]

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "add-dataset-btn" in triggered_id and selected_dataset:
        if selected_dataset not in active_datasets:
            active_datasets.append(selected_dataset)

    elif "remove-dataset-btn" in triggered_id:
        triggered_index = eval(triggered_id)["index"]
        active_datasets = [dataset for dataset in active_datasets if dataset != triggered_index]

    return [
        html.Div([
            dcc.Link(
                dataset,
                href=f"/stream-data",
                style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
            ),
            html.Button("Stop", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
            })
        ]) for dataset in active_datasets
    ]

@callback(
    [Output("popup", "style"), Output("popup-interval", "disabled")],
    [Input("add-dataset-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
    [State("popup", "style")]
)
def handle_popup(n_clicks, n_intervals, style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return style, True

    trigger = ctx.triggered[0]["prop_id"]
    if trigger == "add-dataset-btn.n_clicks" and n_clicks:
        style.update({"display": "block"})
        return style, False 
    elif trigger == "popup-interval.n_intervals":
        style.update({"display": "none"})
        return style, True 

    return style, True

@callback(
        Output("starter-feedback", "children"),
        State("dataset-dropdown", "value"),
        State("detection-model-dropdown", "value"),
        State("mode-selection", "value"),
        State("injection-method-dropdown", "value"),
        State("date-picker-range", "start_date"),
        State("date-picker-range", "end_date"),
        Input("add-dataset-btn", "n_clicks")
        )
def store_current_job_request(selected_dataset, selected_model, selected_mode, selected_injection_method, start_date, end_date, n_clicks):   
        range = (start_date, end_date)
        frontend_handler.user_request(selected_dataset, selected_model, selected_mode, selected_injection_method, range)
    
    
