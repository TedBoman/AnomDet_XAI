from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import random

# Create the Dash app
app = Dash(__name__)

# Mock datasets (these datasets should match the job details from starter_page.py)
datasets = {
    f"Dataset {i}": pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min'),
        "load-15m": [random.uniform(0, 10) for _ in range(100)],
        "cpu-usage": [random.uniform(20, 90) for _ in range(100)],
        "memory-usage": [random.uniform(30, 80) for _ in range(100)],
    }) for i in range(1, 6)
}

# Initial dataset (will be overridden by the selected job)
current_dataset = list(datasets.keys())[0]
anomaly_log = []  # Global anomaly log

# Layout
layout = html.Div([
    # Store component to hold the current dataset and columns selection
    dcc.Store(id="store-data", storage_type="session"),  # Store component for global data

    # Back to Home
    html.Div([html.A("← Back to Home", href="/", style={
        "fontSize": "24px", "color": "#ffffff", "fontWeight": "bold",
        "textDecoration": "none", "position": "absolute", "top": "10px", "left": "20px",
        "padding": "5px 10px", "backgroundColor": "#4CAF50", "borderRadius": "5px",
        "boxShadow": "0px 0px 10px rgba(0,0,0,0.3)"
    })]),

    html.H1("Stream Data Page", style={"textAlign": "center", "color": "#ffffff"}),

    # Left Panel: Column Selection + Anomaly Log
    html.Div([
        html.H3("Available Columns:", style={"color": "#ffffff", "textAlign": "center"}),
        dcc.Checklist(
            id="column-selector", options=[], value=[],
            style={"color": "#ffffff", "padding": "10px", "fontSize": "16px"}
        ),
        html.H3("Anomaly Log:", style={"color": "#ffffff", "textAlign": "center", "marginTop": "20px"}),
        html.Div(id="anomaly-log", style={
            "height": "200px", "overflowY": "scroll", "backgroundColor": "#1e2130",
            "color": "#ffffff", "padding": "10px", "borderRadius": "5px", "border": "1px solid #444"
        })
    ], style={"width": "20%", "float": "left", "backgroundColor": "#1e2130", "padding": "20px", "borderRadius": "10px"}),

    # Right Panel: Graphs
    html.Div(id="selected-graphs", style={"width": "75%", "float": "right", "padding": "20px"}),

    # Interval for streaming
    dcc.Interval(id="stream-interval", interval=1000, n_intervals=0)
], style={"backgroundColor": "#282c34", "padding": "50px", "minHeight": "100vh", "position": "relative"})

# Callbacks
# Callbacks
def register_callbacks(app):
    global anomaly_log

    # Update the store with selected dataset and columns based on URL
    @app.callback(
        Output("store-data", "data"),
        Input("url", "pathname")
    )
    def store_data(pathname):
        """ Store the dataset name and columns selected from the URL """
        dataset_name = pathname.split('/')[-1]
        if dataset_name in datasets:
            columns = datasets[dataset_name].columns.tolist()
            columns.remove("timestamp")
            return {"dataset_name": dataset_name, "columns": columns}
        return {}

    # Update available columns based on the selected dataset
    @app.callback(
        Output("column-selector", "options"),
        Input("store-data", "data")
    )
    def update_column_selector(store_data):
        """ Update available columns based on the stored dataset """
        if store_data and "dataset_name" in store_data:
            dataset_name = store_data["dataset_name"]
            columns = datasets[dataset_name].columns.tolist()
            columns.remove("timestamp")
            return [{"label": col, "value": col} for col in columns]
        return []

    # Update graphs and anomaly log based on stored data
    @app.callback(
        [Output("selected-graphs", "children"),
         Output("anomaly-log", "children")],
        [Input("stream-interval", "n_intervals"),
         Input("store-data", "data"),
         Input("column-selector", "value")]
    )
    def update_graphs_and_anomalies(n_intervals, store_data, selected_columns):
        """ Generate graphs and update anomaly logs based on dataset from URL """
        if not store_data or "dataset_name" not in store_data:
            return html.Div("Dataset not found.", style={"color": "#ffffff", "textAlign": "center"}), []

        dataset_name = store_data["dataset_name"]
        df = datasets[dataset_name]

        # Simulate new streaming data
        new_data = {"timestamp": [pd.Timestamp.now()]}
        for col in df.columns:
            if col != "timestamp":
                new_data[col] = [random.uniform(df[col].min(), df[col].max())]
        datasets[dataset_name] = pd.concat([df, pd.DataFrame(new_data)]).tail(100)

        # Generate graphs for selected columns
        graphs = []
        new_anomalies = []
        for col in selected_columns:
            threshold = df[col].mean() + 2 * df[col].std()
            anomalies = df[df[col] > threshold]

            fig = go.Figure([ 
                go.Scatter(x=df["timestamp"], y=df[col], mode="lines+markers", name=col),
                go.Scatter(x=anomalies["timestamp"], y=anomalies[col], mode="markers",
                           marker=dict(color="red", size=10), name="Anomalies")
            ])
            fig.update_layout(
                title=f"{col.replace('-', ' ').title()} Over Time ({dataset_name})",
                xaxis_title="Timestamp", yaxis_title=col.replace("-", " ").title(),
                template="plotly_dark"
            )
            graphs.append(dcc.Graph(figure=fig, style={"marginBottom": "30px"}))

            for _, row in anomalies.iterrows():
                new_anomalies.append(f"[{row['timestamp']}] {dataset_name} - Anomaly in {col}: {row[col]:.2f}")

        anomaly_log.extend(new_anomalies)
        return graphs, html.Ul([html.Li(log) for log in anomaly_log[-10:]])

# Register the callbacks
register_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

