from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import random

# Sahte veri oluştur
timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min')
data = {
    'timestamp': timestamps,
    'load-15m': [random.uniform(0, 10) for _ in range(100)],
    'cpu-usage': [random.uniform(20, 90) for _ in range(100)],
    'memory-usage': [random.uniform(30, 80) for _ in range(100)],
}
df = pd.DataFrame(data)

layout = html.Div([
    html.H1("Stream Data Page", style={"textAlign": "center"}),
    html.Div("This page shows live streaming data."),

    # Zamanlayıcı
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0),

    # Grafikler
    dcc.Graph(id="load-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="cpu-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="memory-graph", style={"marginBottom": "30px"}),

    html.A("Back to Home", href="/", style={'display': 'block', 'textAlign': 'center', 'marginTop': '20px'})
])


# Callback fonksiyonlarını kaydeden bir fonksiyon
def register_callbacks(app):
    @app.callback(
        Output("load-graph", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_load_graph(n_intervals):
        global df
        new_data = {
            "timestamp": [pd.Timestamp.now()],
            "load-15m": [random.uniform(0, 10)],
            "cpu-usage": [random.uniform(20, 90)],
            "memory-usage": [random.uniform(30, 80)]
        }
        df = pd.concat([df, pd.DataFrame(new_data)]).tail(100)

        # 15m Load grafiği
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["load-15m"],
                mode="lines+markers",
                name="15m Load",
                line=dict(color="blue"),
            )
        ])
        figure.update_layout(
            title="15m Load Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Load (15m)",
            template="plotly_dark",
        )
        return figure

    @app.callback(
        Output("cpu-graph", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_cpu_graph(n_intervals):
        global df
        # CPU Usage grafiği
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["cpu-usage"],
                mode="lines+markers",
                name="CPU Usage",
                line=dict(color="green"),
            )
        ])
        figure.update_layout(
            title="CPU Usage Over Time",
            xaxis_title="Timestamp",
            yaxis_title="CPU Usage (%)",
            template="plotly_dark",
        )
        return figure

    @app.callback(
        Output("memory-graph", "figure"),
        Input("interval-component", "n_intervals")
    )
    def update_memory_graph(n_intervals):
        global df
        # Memory Usage grafiği
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["memory-usage"],
                mode="lines+markers",
                name="Memory Usage",
                line=dict(color="purple"),
            )
        ])
        figure.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Memory Usage (%)",
            template="plotly_dark",
        )
        return figure
