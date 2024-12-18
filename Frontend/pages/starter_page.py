import os
import requests
import socket
import json
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
from dash.dependencies import ALL
BACKEND_HOST = 'Backend'
BACKEND_PORT = int(os.getenv('BACKEND_PORT'))

def send_socket_request(data):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((BACKEND_HOST, BACKEND_PORT))
        sock.sendall(bytes(data, encoding="utf-8"))
        response = sock.recv(1024).decode('utf-8')
        sock.close()
        return json.loads(response) if response else None
    except Exception as e:
        print(f"Socket error: {e}")
        return None

def get_datasets():
    data = json.dumps({"METHOD": "get-datasets"})
    response = send_socket_request(data)
    if "datasets" in response:
        return response["datasets"]
    return []

def get_models():
    data = json.dumps({"METHOD": "get-models"})
    response = send_socket_request(data)
    if "models" in response:
        return response["models"]
    return []

datasets = get_datasets()
models = get_models()

active_datasets = []

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

    html.Div(id="starter-feedback", style={"textAlign": "center", "marginTop": "20px"}),

    html.Div(id="starter-feedback", style={"textAlign": "center", "marginTop": "20px"}),

    # Additional Panel Section
    html.Div(id="additional-panel", style={"marginTop": "20px", "color": "#ffffff"}),

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

#callback to save values of a job that the user is instantiating
#@callback(
#    Output("starter-feedback", "children"),
#    Input("dataset-dropdown", "value"),
#    Input("detection-model-dropdown", "value"),
#    Input("")
    

#def save_user_choices(n_clicks, job_type, selected_dataset, selected_model selected_inj_method):

 #   pass











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


# Callback to show/hide injection panel
@callback(
    Output("injection-panel", "style"),
    Input("injection-check", "value")
)
def toggle_injection_panel(selected):
    if "use_injection" in selected:
        return {"display": "block", "marginTop": "20px", "textAlign": "center"}
    return {"display": "none"}


# Callback to update injection panel content
@callback(
    Output("injection-panel", "children"),
    Input("injection-check", "value")
)
def update_injection_panel(selected):
    if "use_injection" in selected:
        return html.Div([
            html.Label("Select an Injection Method:", style={"fontSize": "22px", "color": "#ffffff"}),
            dcc.Dropdown(
                id="injection-method-dropdown",
                options=[
                    {"label": "Method 1", "value": "method_1"},
                    {"label": "Method 2", "value": "method_2"},
                    {"label": "Method 3", "value": "method_3"}
                ],
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
            ], style={"marginTop": "20px", "textAlign": "center"})
        ])
    return ""