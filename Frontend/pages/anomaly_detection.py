from dash import dcc, html, Input, Output,callback
import pandas as pd
import plotly.graph_objs as go
import random

# Initial fake data
timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min')
data = {
    'timestamp': timestamps,
    'load-15m': [random.uniform(0, 10) for _ in range(100)],
    'cpu-usage': [random.uniform(20, 90) for _ in range(100)],
    'memory-usage': [random.uniform(30, 80) for _ in range(100)],
}
df = pd.DataFrame(data)

layout = html.Div([
    html.H1("Stream Data Page", style={"textAlign": "center", "color": "#ffffff"}),

    html.Div("This page shows live streaming data.", style={"textAlign": "center", "color": "#ffffff"}),

    # Zamanlayıcı
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0),

    # Grafikler
    dcc.Graph(id="load-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="cpu-graph", style={"marginBottom": "30px"}),
    dcc.Graph(id="memory-graph", style={"marginBottom": "30px"}),

    # Anomaly Log (Chatbox-style)
    html.Div([
        html.H3("Anomaly Log:", style={"color": "#ffffff", "marginBottom": "10px"}),
        html.Div(id="anomaly-log", style={
            "height": "150px",
            "overflowY": "scroll",
            "backgroundColor": "#1e2130",
            "color": "#ffffff",
            "padding": "10px",
            "border": "1px solid #444",
            "borderRadius": "5px"
        })
    ], style={"position": "fixed", "bottom": "10px", "left": "10px", "width": "300px"}),

    html.A("Back to Home", href="/", style={'display': 'block', 'textAlign': 'center', 'marginTop': '20px', 'color': '#ffffff'})
], style={"backgroundColor": "#282c34", "padding": "50px", "minHeight": "100vh"})


# Global dataframe for updates
global_df = df.copy()

# Callback to update Load Graph
@callback(
    Output("load-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_load_graph(n_intervals):
    global global_df

    # Update fake data
    new_data = {
        "timestamp": [pd.Timestamp.now()],
        "load-15m": [random.uniform(0, 10)],
        "cpu-usage": [random.uniform(20, 90)],
        "memory-usage": [random.uniform(30, 80)]
    }
    global_df = pd.concat([global_df, pd.DataFrame(new_data)]).tail(100)

    # Identify anomalies
    global_df["load-anomaly"] = (global_df["load-15m"] > 9)

    figure = go.Figure([
        go.Scatter(
            x=global_df["timestamp"],
            y=global_df["load-15m"],
            mode="lines",
            name="15m Load",
            line=dict(color="blue"),
        ),
        go.Scatter(
            x=global_df[global_df["load-anomaly"]]["timestamp"],
            y=global_df[global_df["load-anomaly"]]["load-15m"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomalies",
        )
    ])
    figure.update_layout(
        title="15m Load Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Load (15m)",
        template="plotly_dark",
    )
    return figure


# Callback to update CPU Graph
@callback(
    Output("cpu-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_cpu_graph(n_intervals):
    global global_df

    # Identify anomalies
    global_df["cpu-anomaly"] = (global_df["cpu-usage"] > 70)

    figure = go.Figure([
        go.Scatter(
            x=global_df["timestamp"],
            y=global_df["cpu-usage"],
            mode="lines",
            name="CPU Usage",
            line=dict(color="green"),
        ),
        go.Scatter(
            x=global_df[global_df["cpu-anomaly"]]["timestamp"],
            y=global_df[global_df["cpu-anomaly"]]["cpu-usage"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomalies",
        )
    ])
    figure.update_layout(
        title="CPU Usage Over Time",
        xaxis_title="Timestamp",
        yaxis_title="CPU Usage (%)",
        template="plotly_dark",
    )
    return figure


# Callback to update Memory Graph
@callback(
    Output("memory-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_memory_graph(n_intervals):
    global global_df

    # Identify anomalies
    global_df["memory-anomaly"] = (global_df["memory-usage"] > 80)

    figure = go.Figure([
        go.Scatter(
            x=global_df["timestamp"],
            y=global_df["memory-usage"],
            mode="lines",
            name="Memory Usage",
            line=dict(color="purple"),
        ),
        go.Scatter(
            x=global_df[global_df["memory-anomaly"]]["timestamp"],
            y=global_df[global_df["memory-anomaly"]]["memory-usage"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Anomalies",
        )
    ])
    figure.update_layout(
        title="Memory Usage Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Memory Usage (%)",
        template="plotly_dark",
    )
    return figure


# Callback to update Anomaly Log
@callback(
    Output("anomaly-log", "children"),
    Input("interval-component", "n_intervals")
)
def update_anomaly_log(n_intervals):
    global global_df

    # Collect anomalies
    anomalies = []
    for index, row in global_df.iterrows():
        if row.get("load-anomaly"):
            anomalies.append(f"{row['timestamp']} - Load anomaly detected: {row['load-15m']:.2f}")
        if row.get("cpu-anomaly"):
            anomalies.append(f"{row['timestamp']} - CPU anomaly detected: {row['cpu-usage']:.2f}")
        if row.get("memory-anomaly"):
            anomalies.append(f"{row['timestamp']} - Memory anomaly detected: {row['memory-usage']:.2f}")

    return [html.Div(anomaly) for anomaly in anomalies[-10:]]  # Display the last 10 anomalies
