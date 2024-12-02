from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from stream_data import df 
# Sahte veri kullanımı için Stream Page'deki dataframe'i alıyoruz
from stream_data import df

layout = html.Div([
    html.H1("Anomaly Detection", style={"textAlign": "center", "marginBottom": "30px", "color": "#ffffff"}),

    # Anomaly Threshold Ayarı
    html.Div([
        html.Label("Anomaly Threshold:", style={"color": "#ffffff", "fontSize": "18px"}),
        dcc.Slider(
            id="threshold-slider",
            min=0,
            max=100,
            step=1,
            value=8,  # Varsayılan eşik
            marks={i: str(i) for i in range(0, 101, 10)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"marginBottom": "30px", "padding": "20px"}),

    # Anomaly Tespit Edilen Grafikler
    dcc.Graph(id="load-anomaly-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="cpu-anomaly-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="memory-anomaly-graph", style={"marginBottom": "30px"}),

    html.A("Back to Home", href="/", style={'display': 'block', 'textAlign': "center", "marginTop": "20px"})
], style={
    "backgroundColor": "#282c34",
    "padding": "50px",
    "minHeight": "100vh"
})


def register_callbacks(app):
    @app.callback(
        Output("load-anomaly-graph", "figure"),
        [Input("threshold-slider", "value")]
    )
    def update_load_anomaly(threshold):
        # Anomaly kriterleri
        df["load-anomaly"] = (df["load-15m"] > 9) | (df["load-15m"] < 1)
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["load-15m"],
                mode="lines",
                name="15m Load",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=df[df["load-anomaly"]]["timestamp"],
                y=df[df["load-anomaly"]]["load-15m"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Anomalies",
            )
        ])
        figure.update_layout(
            title="15m Load Anomalies",
            xaxis_title="Timestamp",
            yaxis_title="15m Load",
            template="plotly_dark",
        )
        return figure

    @app.callback(
        Output("cpu-anomaly-graph", "figure"),
        [Input("threshold-slider", "value")]
    )
    def update_cpu_anomaly(threshold):
        # Anomaly kriterleri
        df["cpu-anomaly"] = (df["cpu-usage"] > 70) | (df["cpu-usage"] < 30)
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["cpu-usage"],
                mode="lines",
                name="CPU Usage",
                line=dict(color="green"),
            ),
            go.Scatter(
                x=df[df["cpu-anomaly"]]["timestamp"],
                y=df[df["cpu-anomaly"]]["cpu-usage"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Anomalies",
            )
        ])
        figure.update_layout(
            title="CPU Usage Anomalies",
            xaxis_title="Timestamp",
            yaxis_title="CPU Usage (%)",
            template="plotly_dark",
        )
        return figure

    @app.callback(
        Output("memory-anomaly-graph", "figure"),
        [Input("threshold-slider", "value")]
    )
    def update_memory_anomaly(threshold):
        # Anomaly kriterleri
        df["memory-anomaly"] = (df["memory-usage"] > 70) | (df["memory-usage"] < 35)
        figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["memory-usage"],
                mode="lines",
                name="Memory Usage",
                line=dict(color="purple"),
            ),
            go.Scatter(
                x=df[df["memory-anomaly"]]["timestamp"],
                y=df[df["memory-anomaly"]]["memory-usage"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Anomalies",
            )
        ])
        figure.update_layout(
            title="Memory Usage Anomalies",
            xaxis_title="Timestamp",
            yaxis_title="Memory Usage (%)",
            template="plotly_dark",
        )
        return figure
