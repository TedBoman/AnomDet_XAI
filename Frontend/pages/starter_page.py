from dash import dcc, html, Input, Output, callback

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

    # Detection Model Dropdown
    html.Div([
        html.Label("Select a Detection Model:", style={"fontSize": "22px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="model-dropdown",
            options=[
                {"label": "Model A", "value": "model_a"},
                {"label": "Model B", "value": "model_b"},
                {"label": "Model C", "value": "model_c"}
            ],
            placeholder="Select a model",
            style={"width": "350px", "fontSize": "18px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    # Injection Checkbox and Panel
    html.Div([
        html.Div([
            dcc.Checklist(
                id="injection-check",
                options=[{"label": "Use Injection", "value": "use_injection"}],
                value=[],
                style={"textAlign": "center", "fontSize": "20px", "color": "#ffffff"}
            )
        ], style={"textAlign": "center"}),

        html.Div(id="injection-panel", style={"display": "none", "marginTop": "20px", "textAlign": "center"})
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    html.Div(id="starter-feedback", style={"textAlign": "center", "marginTop": "20px"}),

    # Logo Section at the Bottom
  
], style={
    "backgroundColor": "#282c34",  # Background color
    "padding": "50px",
    "minHeight": "100vh"
})


# Callback to Show/Hide Injection Panel
@callback(
    Output("injection-panel", "style"),
    Input("injection-check", "value")
)
def toggle_injection_panel(selected):
    if "use_injection" in selected:
        return {"display": "block", "marginTop": "20px", "textAlign": "center"}
    return {"display": "none"}


# Callback to Add Dropdown and Date Range to Injection Panel
@callback(
    Output("injection-panel", "children"),
    Input("injection-check", "value")
)
def render_injection_panel(selected):
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
                placeholder="Select an injection method",
                style={"width": "350px", "fontSize": "18px", "margin": "auto"}
            ),
            html.Div([
                html.Label("Select Date Range for Injection:", style={"fontSize": "18px", "color": "#ffffff"}),
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
