from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import plotly.graph_objs as go
import random

# Mock datasets
datasets = {
    f"Dataset {i}": pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min'),
        "load-15m": [random.uniform(0, 10) for _ in range(100)],
        "cpu-usage": [random.uniform(20, 90) for _ in range(100)],
        "memory-usage": [random.uniform(30, 80) for _ in range(100)],
    }) for i in range(1, 6)
}

current_dataset = list(datasets.keys())[0]
anomaly_log = []  # Global anomaly log

# Layout
layout = html.Div([
    # Back to Home
    html.Div([
        html.A("← Back to Home", href="/", style={
            "fontSize": "24px", "color": "#ffffff", "fontWeight": "bold", 
            "textDecoration": "none", "position": "absolute", "top": "10px", 
            "left": "20px", "padding": "5px 10px", "backgroundColor": "#4CAF50",
            "borderRadius": "5px", "boxShadow": "0px 0px 10px rgba(0,0,0,0.3)"
        })
    ]),

    html.H1("Stream Data Page", style={"textAlign": "center", "color": "#ffffff"}),

    # Dataset Selection
    html.Div([
        html.Label("Select Dataset:", style={"fontSize": "20px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="stream-dataset-dropdown",
            options=[{"label": name, "value": name} for name in datasets.keys()],
            value=current_dataset,
            style={"width": "300px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Left Panel: Column Selection + Anomaly Log
    html.Div([
        html.H3("Available Columns:", style={"color": "#ffffff", "textAlign": "center"}),
        dcc.Checklist(
            id="column-selector",
            options=[], value=[],
            style={"color": "#ffffff", "padding": "10px", "fontSize": "16px"}
        ),
        html.H3("Anomaly Log:", style={"color": "#ffffff", "textAlign": "center", "marginTop": "20px"}),
        html.Div(id="anomaly-log", style={
            "height": "200px", "overflowY": "scroll", "backgroundColor": "#1e2130",
            "color": "#ffffff", "padding": "10px", "borderRadius": "5px",
            "border": "1px solid #444"
        })
    ], style={"width": "20%", "float": "left", "backgroundColor": "#1e2130",
              "padding": "20px", "borderRadius": "10px"}),

    # Right Panel: Graphs
    html.Div(id="selected-graphs", style={
        "width": "75%", "float": "right", "padding": "20px"
    }),

    # Interval for streaming
    dcc.Interval(id="stream-interval", interval=1000, n_intervals=0)
], style={"backgroundColor": "#282c34", "padding": "50px", "minHeight": "100vh", "position": "relative"})


# Callbacks
def register_callbacks(app):
    global anomaly_log

    @app.callback(
        Output("column-selector", "options"),
        Input("stream-dataset-dropdown", "value")
    )
    def update_column_selector(selected_dataset):
        """ Update available columns based on selected dataset """
        columns = datasets[selected_dataset].columns.tolist()
        columns.remove("timestamp")
        return [{"label": col, "value": col} for col in columns]

    @app.callback(
        [Output("selected-graphs", "children"),
         Output("anomaly-log", "children")],
        [Input("stream-interval", "n_intervals"),
         Input("stream-dataset-dropdown", "value"),
         Input("column-selector", "value")]
    )
    def update_graphs_and_anomalies(n_intervals, selected_dataset, selected_columns):
        """ Generate graphs and update anomaly logs """
        if not selected_columns:
            return html.Div("No columns selected.", style={"color": "#ffffff", "textAlign": "center"}), []

        global anomaly_log
        df = datasets[selected_dataset]

        # Simulate new streaming data
        new_data = {"timestamp": [pd.Timestamp.now()]}
        for col in df.columns:
            if col != "timestamp":
                new_data[col] = [random.uniform(df[col].min(), df[col].max())]
        datasets[selected_dataset] = pd.concat([df, pd.DataFrame(new_data)]).tail(100)

        graphs = []
        new_anomalies = []
        threshold={
            "load-15m":8,
            "cpu-usage":85,
            "memory-usage":75
        }

        for col in selected_columns:
            # Detect anomalies
            threshold = df[col].mean() + 2 * df[col].std()
            anomalies = df[df[col] > threshold]

            # Plot graph
            fig = go.Figure([
                go.Scatter(
                    x=df["timestamp"], y=df[col], mode="lines+markers",
                    name=col, marker=dict(size=5, color="blue")
                ),
                go.Scatter(
                    x=anomalies["timestamp"], y=anomalies[col], mode="markers",
                    marker=dict(color="red", size=10), name="Anomalies"
                )
            ])
            fig.update_layout(
                title=f"{col.replace('-', ' ').title()} Over Time ({selected_dataset})",
                xaxis_title="Timestamp", yaxis_title=col.replace("-", " ").title(),
                template="plotly_dark"
            )
            graphs.append(dcc.Graph(figure=fig, style={"marginBottom": "30px"}))

            # Add anomalies to log
            for _, row in anomalies.iterrows():
                new_anomalies.append(f"[{row['timestamp']}] {selected_dataset} - Anomaly in {col}: {row[col]:.2f}")

        anomaly_log.extend(new_anomalies)
        return graphs, html.Ul([html.Li(log) for log in anomaly_log[-10:]])