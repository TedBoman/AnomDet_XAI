from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
import random

# Sahte veri üretme
def generate_fake_data():
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='min')
    data = {
        'timestamp': timestamps,
        'load-15m': [random.uniform(0, 10) for _ in range(100)],
        'cpu-usage': [random.uniform(20, 90) for _ in range(100)],
        'memory-usage': [random.uniform(30, 80) for _ in range(100)]
    }
    return pd.DataFrame(data)

df = generate_fake_data()

layout = html.Div([
    html.H1("Load Data Page", style={"textAlign": "center"}),
    html.Div("This page shows all loaded data."),
    dcc.Graph(
        figure={
            "data": [
                go.Table(
                    header={"values": list(df.columns)},
                    cells={"values": [df[col] for col in df.columns]}
                )
            ]
        }
    ),
    html.A("Back to Home", href="/", style={'display': 'block', 'textAlign': 'center', 'marginTop': '20px'})
])
