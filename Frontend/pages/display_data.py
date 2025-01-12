from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
import random
from datetime import datetime, timezone

graphs = {}
def create_graphs(df, columns):
    global graphs
    points_per_frame = 500
    print(df)
    
    if len(df) == 1:
        x_min = -1
        x_max = 1
    elif len(df) <= points_per_frame:
        x_min = df["timestamp"].min() * 0.9
        x_max = df["timestamp"].max() * 1.1
    else:
        max_time = df["timestamp"].max()
        while len(df[df["timestamp"] < max_time]) > points_per_frame:
            max_time *= 0.9
        x_min = df["timestamp"].min() * 0.9
        x_max = max_time
    x_range = (x_min, x_max)

    for col in columns:
        y = df[col]
        y_min = df[col].astype("float32").min()
        y_max = df[col].astype("float32").max()
        y_range = (y_min*0.8, y_max*1.2)

        true_normal = df[(df["is_anomaly"] == False) & (df["injected_anomaly"] == False)][["timestamp", col]]
        false_normal = df[(df["is_anomaly"] == False) & (df["injected_anomaly"] == True)][["timestamp", col]]
        anomalies = df[df["is_anomaly"] == True][["timestamp", col]]

        p = figure(
            width=1400, 
            height=350, 
            title=f"{col} timeline", 
            x_axis_label="Time", 
            y_axis_label=col, 
            x_range=x_range,
            y_range=y_range,
            tools="pan,reset,save",
        )

        p.scatter(true_normal["timestamp"], true_normal[col], size=6, color="green", alpha=0.7, legend_label="Normal Data")
        if len(false_normal) > 0:
            p.scatter(false_normal["timestamp"], false_normal[col], size=6, color="blue", alpha=0.7, legend_label="Injected Anomalies Labeled as Normal", marker="diamond")  
        if len(anomalies) > 0:
            p.scatter(anomalies["timestamp"], anomalies[col], size=6, color="red", alpha=0.7, legend_label="Anomalies", marker="x")

        p.legend.location = "top_right"
        p.x_range.bounds = (x_min, x_max)
        p.y_range.bounds = "auto"

        html_content = file_html(p, CDN, f"{col} Plot")
        graphs[col] = html.Iframe(srcDoc=html_content, style={"width": "100%", "height": "370px", "border": "none"})

def create_default_columns(columns):
    if len(columns) > 3:
        return [columns[0], columns[1], columns[2]]
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