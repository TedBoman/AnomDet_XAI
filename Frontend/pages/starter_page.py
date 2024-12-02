from dash import dcc, html, Input, Output, callback
import dash

layout = html.Div([
    # Sayfa Başlığı
    html.H1("Starter Page", style={
        "textAlign": "center", 
        "marginBottom": "30px", 
        "color": "#ffffff", 
        "fontSize": "40px"}),

    # Load ve Stream Data Butonları
    html.Div([
        html.Button("Load Data", id="load-data-btn", n_clicks=0, 
                    style={"margin": "10px", "width": "300px", "height": "70px", 
                           "fontSize": "20px", "backgroundColor": "#4CAF50", 
                           "color": "#ffffff", "borderRadius": "10px"}),
        html.Button("Stream Data", id="stream-data-btn", n_clicks=0, 
                    style={"margin": "10px", "width": "300px", "height": "70px", 
                           "fontSize": "20px", "backgroundColor": "#008CBA", 
                           "color": "#ffffff", "borderRadius": "10px"})
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

    # Injection Checkbox ve Panel
    html.Div([
        # Injection Checkbox
        html.Div([
            dcc.Checklist(
                id="injection-check",
                options=[{"label": "Use Injection", "value": "use_injection"}],
                value=[],
                style={"textAlign": "center", "fontSize": "20px", "color": "#ffffff"}
            )
        ], style={"textAlign": "center"}),

        # Injection Model Panel (başlangıçta gizli)
        html.Div(id="injection-panel", style={"display": "none", "marginTop": "20px", "textAlign": "center"})
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    # Feedback Paneli (Sonuç veya Mesajlar)
    html.Div(id="starter-feedback", style={"textAlign": "center", "marginTop": "20px"})
], style={
    "backgroundColor": "#282c34",  # Arka plan rengi
    "padding": "50px",
    "minHeight": "100vh"
})


# Callback: Injection Panelini Göster/Gizle
@callback(
    Output("injection-panel", "style"),
    Input("injection-check", "value")
)
def toggle_injection_panel(selected):
    if "use_injection" in selected:
        return {"display": "block", "marginTop": "20px", "textAlign": "center"}
    return {"display": "none"}

# Callback: Injection Paneline Dropdown Ekleyin
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
            )
        ])
    return ""

# Callback: Load Data ve Stream Data Butonlarını Yönet
@callback(
    Output("starter-feedback", "children"),
    [Input("load-data-btn", "n_clicks"),
     Input("stream-data-btn", "n_clicks")]
)
def navigate_buttons(load_clicks, stream_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "load-data-btn":
        return dcc.Location(href="/load-data", id="redirect-load-data")
    elif button_id == "stream-data-btn":
        return dcc.Location(href="/stream-data", id="redirect-stream-data")
    return ""
