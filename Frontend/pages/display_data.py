from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
import random
from datetime import datetime, timezone

graphs = {}
def create_graphs(df, columns):
    global graphs
    i = 1
    for col in columns:
        normal = df[df["is_anomaly"] == False][["timestamp", col]]
        anomalies = df[df["is_anomaly"] == True][["timestamp", col]]
        fig_layout = go.Layout(
            width=800,  # Width of the figure
            height=400  # Height of the figure
        )
        fig = go.Figure(
                data=[
                    go.Scatter(x=normal["timestamp"], y=normal[col], mode="markers", name=col),
                    go.Scatter(x=anomalies["timestamp"], y=anomalies[col], mode="markers", marker = dict(color="red", size=10), name="Anomalies")
                ],
                layout=fig_layout
            )
        fig.update_layout(title=col, xaxis_title="Time", yaxis_title=col)
        graph = dcc.Graph(id = {"type" : "graph", "index" : col}, figure = fig, style={"padding": "15px"})
        graphs[col] = graph

def create_default_columns(columns):
    if len(columns) > 3:
        return random.sample(columns, 3)
    return columns

def layout(handler, job_name, batch=True):
    #Get data frame from a completed job
    df = handler.handle_get_data(0, job_name)

    
    #Create graphs of each column in that data frame
    columns = df.columns.tolist()
    columns.remove("timestamp")
    columns.remove("is_anomaly")
    columns.remove("injected_anomaly")
    create_graphs(df, columns)

    columns_to_show = create_default_columns(columns)

    # Layout
    layout = html.Div([
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
            dcc.Dropdown(
                id="graph-dropdown",
                options=[{"label": col, "value": col} for col in columns],
                multi=True,
                value=columns_to_show,
                placeholder="Select a method",
                style={"width": "350px", "margin": "auto"}
            ),
        ], style={"textAlign": "center", "padding": "20px", "borderRadius": "10px"}),

        # Right Panel: Graphs
        html.Div(children=[graphs[graph] for graph in columns_to_show], id="graph-container", style={"display": "flex", "justify-content": "center", "flex-direction": "column","padding": "20px", "width": "100%"}),

        # Interval for streaming
        dcc.Interval(id="stream-interval", interval=1000, n_intervals=0, disabled=batch)
        
    ], style={"display": "flex", "justify-content": "center", "flex-direction": "column", "backgroundColor": "#282c34", "width": "100%"})    

    return layout   

def get_local_callback(app):
    # Register callback for updating graphs
    @app.callback(
        Output('graph-container', 'children'),
        [Input('graph-dropdown', 'value')]
    )
    def update_graphs(selected_graphs):
        print("Selected Graphs: ", selected_graphs)
        if not selected_graphs:
            return [[]]
        return [graphs[graph] for graph in selected_graphs]