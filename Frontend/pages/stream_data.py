from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
import random

# Generate fake data
def generate_fake_data():
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min')
    data = {
        'timestamp': timestamps,
        'load-15m': [random.uniform(0, 10) for _ in range(100)],
        'cpu-usage': [random.uniform(20, 90) for _ in range(100)],
        'memory-usage': [random.uniform(30, 80) for _ in range(100)],
    }
    return pd.DataFrame(data)

# Local dataframe for Stream Data
df = generate_fake_data()

# Layout for Stream Data Page
layout = html.Div([
    html.H1("Stream Data Page", style={"textAlign": "center", "color": "#ffffff"}),

    html.Div("This page shows live streaming data.", style={"textAlign": "center", "color": "#ffffff"}),

    # Interval for updating data
    dcc.Interval(id="stream-interval", interval=1000, n_intervals=0),

    # Graphs
    dcc.Graph(id="stream-load-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="stream-cpu-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="stream-memory-graph", style={"marginBottom": "30px"}),

    # Log Box for Anomalies
    html.Div([
        html.H3("Anomaly Log:", style={"color": "#ffffff", "marginBottom": "10px"}),
        html.Div(id="stream-anomaly-log", style={
            "height": "200px",
            "overflowY": "scroll",
            "backgroundColor": "#1e2130",
            "color": "#ffffff",
            "padding": "10px",
            "border": "1px solid #444",
            "borderRadius": "5px"
        })
    ], style={"marginTop": "30px"}),

    html.A("Back to Home", href="/", style={'display': 'block', 'textAlign': 'center', 'marginTop': '20px', 'color': '#ffffff'})
], style={"backgroundColor": "#282c34", "padding": "50px", "minHeight": "100vh"})


# Register callbacks for Stream Data
def register_callbacks(app):
    anomaly_log = []  # Store anomalies here

    @app.callback(
        [
            Output("stream-load-graph", "figure"),
            Output("stream-cpu-graph", "figure"),
            Output("stream-memory-graph", "figure"),
            Output("stream-anomaly-log", "children"),
        ],
        Input("stream-interval", "n_intervals")
    )
    def update_stream_data(n_intervals):
        global df  # Use the local dataframe instance

        # Generate new data
        new_data = {
            "timestamp": [pd.Timestamp.now()],
            "load-15m": [random.uniform(0, 10)],
            "cpu-usage": [random.uniform(20, 90)],
            "memory-usage": [random.uniform(30, 80)],
        }
        df = pd.concat([df, pd.DataFrame(new_data)]).tail(100)

        # Detect anomalies
        load_anomalies = df[df["load-15m"] > 9]
        cpu_anomalies = df[df["cpu-usage"] > 85]
        memory_anomalies = df[df["memory-usage"] > 75]

        # Append anomaly messages
        for _, row in load_anomalies.iterrows():
            anomaly_log.append(f"[{row['timestamp']}] High Load: {row['load-15m']:.2f}")
        for _, row in cpu_anomalies.iterrows():
            anomaly_log.append(f"[{row['timestamp']}] High CPU Usage: {row['cpu-usage']:.2f}")
        for _, row in memory_anomalies.iterrows():
            anomaly_log.append(f"[{row['timestamp']}] High Memory Usage: {row['memory-usage']:.2f}")

        # Update graphs
        load_figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["load-15m"],
                mode="lines+markers",
                name="15m Load",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=load_anomalies["timestamp"],
                y=load_anomalies["load-15m"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Load Anomalies",
            )
        ])
        load_figure.update_layout(
            title="15m Load Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Load (15m)",
            template="plotly_dark",
        )

        cpu_figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["cpu-usage"],
                mode="lines+markers",
                name="CPU Usage",
                line=dict(color="green"),
            ),
            go.Scatter(
                x=cpu_anomalies["timestamp"],
                y=cpu_anomalies["cpu-usage"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="CPU Anomalies",
            )
        ])
        cpu_figure.update_layout(
            title="CPU Usage Over Time",
            xaxis_title="Timestamp",
            yaxis_title="CPU Usage (%)",
            template="plotly_dark",
        )

        memory_figure = go.Figure([
            go.Scatter(
                x=df["timestamp"],
                y=df["memory-usage"],
                mode="lines+markers",
                name="Memory Usage",
                line=dict(color="purple"),
            ),
            go.Scatter(
                x=memory_anomalies["timestamp"],
                y=memory_anomalies["memory-usage"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Memory Anomalies",
            )
        ])
        memory_figure.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Memory Usage (%)",
            template="plotly_dark",
        )

        # Update anomaly log (show last 10 entries)
        anomaly_log_display = html.Ul([html.Li(log) for log in anomaly_log[-10:]])
        return load_figure, cpu_figure, memory_figure, anomaly_log_display
