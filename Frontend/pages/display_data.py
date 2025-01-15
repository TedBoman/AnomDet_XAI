from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
import random
from datetime import datetime, timezone
import os
from time import sleep

graphs = {}

def create_default_columns(columns):
    if len(columns) > 3:
        return [columns[0], columns[1], columns[2]]
    return columns

def layout(handler, job_name, batch=True):
    # Get columns in the graph
    columns = handler.handle_get_columns(job_name)
    columns.remove("timestamp")

    # Load graphs from the graphs folder associated with the jobs
    load_graphs(job_name, columns)
    print("THIS WORKS")

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
        html.H3("Available Columns:", style={"color": "#ffffff", "textAlign": "center"}),
        dcc.Checklist(
            id="graph-checklist",
            options=[{"label": col, "value": col} for col in columns],
            value=columns_to_show,
            inline=True,
            style={"width": "1000px", "color": "#ffffff"}
        ),

        # Right Panel: Graphs
        html.Div(children=[graphs[graph] for graph in columns_to_show], id="graph-container", style={"width": "1410px",}),

        # Interval for streaming
        dcc.Interval(id="stream-interval", interval=1000, n_intervals=0, disabled=batch)
        
    ], style={"display": "flex", "align-items": "center", "flex-direction": "column", "backgroundColor": "#282c34", "width": "100%", "minHeight": "100vh"})    

    return layout   

def load_graphs(job_name, columns):
    global graphs
    directory = f"./graphs/{job_name}"

    while not os.path.exists(directory):
        print("Graph is being generated...")
        sleep(2)

    for col in columns:
        file_handle = open(f"{directory}/{col}.html", "r")
        html_content = file_handle.read()
        file_handle.close()
        graphs[col] = html.Iframe(srcDoc=html_content, style={"width": "100%", "height": "370px", "border": "none"})

def get_local_callback(app):
    # Register callback for updating graphs
    @app.callback(
        Output('graph-container', 'children'),
        [Input('graph-checklist', 'value')]
    )
    def update_graphs(selected_graphs):
        if not selected_graphs:
            return [[]]
        return [graphs[graph] for graph in selected_graphs]