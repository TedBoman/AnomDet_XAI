from dash import dcc, html, Input, Output, State, callback, ctx
import dash
import action_handler
from dash.dependencies import ALL

# Placeholder for active datasets
active_datasets = []

# Variables to store selected detection model and injection method
selected_detection_model = None
selected_injection_method = None
selected_model = None

# Placeholder for active datasets
active_datasets = []

layout = html.Div([
    html.H1("Starter Page", style={
        "textAlign": "center",
        "marginBottom": "30px",
        "color": "#ffffff",
        "fontSize": "40px"
    }),

    # Load and Stream Data Buttons
    html.Div([
        dcc.Link(html.Button("Load Data", id="load-data-btn",
                             style={"margin": "10px", "width": "300px", "height": "70px",
                                    "fontSize": "20px", "backgroundColor": "#4CAF50",
                                    "color": "#ffffff", "borderRadius": "10px"}), href="/load-data"),
        dcc.Link(html.Button("Stream Data", id="stream-data-btn",
                             style={"margin": "10px", "width": "300px", "height": "70px",
                                    "fontSize": "20px", "backgroundColor": "#008CBA",
                                    "color": "#ffffff", "borderRadius": "10px"}), href="/stream-data")
    ], style={"textAlign": "center", "marginBottom": "30px"}),
       # Select Detection Model Panel
    html.Div([
        html.Label("Select a Detection Model:", style={"fontSize": "22px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="detection-model-dropdown",
            options=[
                {"label": "Model A", "value": "model_a"},
                {"label": "Model B", "value": "model_b"},
                {"label": "Model C", "value": "model_c"}
            ],
            placeholder="Select a detection model",
            style={"width": "350px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginTop": "30px"}),

    html.Div(id="starter-feedback", style={"textAlign": "center", "marginTop": "20px"}),


    # Select Dataset to Activate Section
    html.Div([
        html.Label("Select Dataset to Activate:", style={"fontSize": "22px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": f"Dataset {i}", "value": f"dataset_{i}"} for i in range(1, 11)],
            placeholder="Select a dataset",
            style={"width": "350px", "fontSize": "18px", "margin": "auto"}
        ),
        html.Button("Add Dataset", id="add-dataset-btn", style={
            "marginTop": "10px", "width": "150px", "height": "40px", "fontSize": "16px",
            "backgroundColor": "#4CAF50", "color": "#ffffff", "borderRadius": "5px"
        })
    ], style={"textAlign": "center", "marginBottom": "30px"}),




    # Active Datasets Section
    html.Div([
        html.H3("Active Datasets:", style={"color": "#ffffff", "textAlign": "center"}),
        html.Div(id="active-datasets-list", style={
            "textAlign": "center", "color": "#ffffff", "marginTop": "10px",
            "padding": "10px", "border": "1px solid #444", "borderRadius": "5px"
        })
    ], style={"marginTop": "30px"}),

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


    dcc.Store(id='selected_dataset'),
    dcc.Store(id='selected_model'),
    dcc.Store(id='selected_inj_method'),


], style={
    "backgroundColor": "#282c34",
    "padding": "50px",
    "minHeight": "100vh"
})


#TO DO: Add callbacks to store values of user choices

#Add callbacks to manage active datasets
@callback(
    Output('selected_dataset', 'data'),
    Input('dataset-dropdown', 'value')
)
def store_selected_dataset(selected_dataset):
    return selected_dataset

#callback to store selected detection model
@callback(
    Output('selected_model', 'data'),
    Input('detection-model-dropdown', 'value')
)
def store_selected_model(selected_model):
    return selected_model

#callback to store selected injection method
@callback(Output('selected_inj_method', 'data'),
          Input('injection-check', 'value')
)
def store_selected_inj_method(selected_inj_method):
    if "use_injection" in selected_inj_method:
        return selected_inj_method
    return None

#Callback to use stored values
@callback(
    Output('starter-feedback', 'children'),  
    [Input('selected_dataset', 'data'),
     Input('selected_model', 'data'),
     Input('selected_inj_method', 'data')]
)
def get_values(selected_dataset, selected_model, selected_inj_method):
    action_handler.user_request(selected_dataset, selected_model, selected_inj_method)
    return ""

    

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
                html.Span(dataset, style={"marginRight": "10px"}),
                html.Button("Remove", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                    "fontSize": "14px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                    "borderRadius": "5px", "padding": "5px"
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
            html.Span(dataset, style={"marginRight": "10px"}),
            html.Button("Remove", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                "fontSize": "14px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                "borderRadius": "5px", "padding": "5px"
            })
        ]) for dataset in active_datasets
    ]


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

