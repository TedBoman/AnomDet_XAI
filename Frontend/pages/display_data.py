from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
import random

'''
    go.Scatter(x=df["timestamp"], y=df[col], mode="lines+markers", name=col),
    go.Scatter(x=anomalies["timestamp"], y=anomalies[col], mode="markers",
                marker=dict(color="red", size=10), name="Anomalies")'''

graphs = {}
def create_graphs(df):
    global graphs
    for col in df.columns.tolist():
        if col != "timestamp":
            anomalies = df[df["is_anomaly"] == True][["timestamp", col]]
            fig = go.Figure([
                go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col),
                go.Scatter(x=anomalies["timestamp"], y=anomalies[col], mode="markers", marker = dict(color="red", size=10), name="Anomalies")
            ])
            fig.update_layout(title=f"{col} over Time", xaxis_title="Time", yaxis_title=col)
            graph = dcc.Graph(id = {"type" : "graph", "index" : col}, figure = fig)
            graphs[col] = graph


'''
# Mock datasets (these datasets should match the job details from starter_page.py)
datasets = {
    f"Dataset {i}": pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min'),
        "load-15m": [random.uniform(0, 10) for _ in range(100)],
        "cpu-usage": [random.uniform(20, 90) for _ in range(100)],
        "memory-usage": [random.uniform(30, 80) for _ in range(100)],
    }) for i in range(1, 6)
}
'''
# Initial dataset (will be overridden by the selected job)
current_dataset = list(datasets.keys())[0]
anomaly_log = []  # Global anomaly log

def layout(handler):
    #Get data frame from a completed job
    df = handler.get_data()
    # Create graphs of each column in that data frame
    create_graphs(df)


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


    @callback(
        Output('graph-container', 'children'),
        [Input('graph-dropdown', 'value')]
    )
    def update_graphs(selected_graphs):
        if not selected_graphs:
            return "Select graphs from the dropdown to display them."
        return [graphs[graph] for graph in selected_graphs]

    return layout   

