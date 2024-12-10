from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import plotly.graph_objs as go
import io
import base64
import random

# Placeholder for datasets
datasets = {
    f"Dataset {i}": pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=100, freq="min"),
        "value": [random.uniform(0, 10) for _ in range(100)]
    }) for i in range(1, 20)
}

# Initial selected dataset columns
dataset_columns = {"Dataset 1": ["value"]}

layout = html.Div([
    html.H1("Load Data Page", style={"textAlign": "center", "color": "#ffffff"}),

    # Dataset selection dropdown
    html.Div([
        html.Label("Select Dataset:", style={"fontSize": "20px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": name, "value": name} for name in datasets.keys()],
            value="Dataset 1",
            style={"width": "300px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Column selection dropdown (updated dynamically)
    html.Div([
        html.Label("Select Column to Plot:", style={"fontSize": "20px", "color": "#ffffff"}),
        dcc.Dropdown(
            id="column-dropdown",
            placeholder="Select a column",
            style={"width": "300px", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Graph to display dataset
    dcc.Graph(id="dataset-graph", style={"marginTop": "20px"}),

    # File upload panel below the graph
    html.Div([
        html.Label("Upload a New Dataset:", style={"fontSize": "20px", "color": "#ffffff"}),
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                "Drag and Drop or ",
                html.A("Select a File", style={"color": "#007BFF", "cursor": "pointer"})
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "20px",
                "backgroundColor": "#f9f9f9"
            },
            multiple=False
        )
    ], style={"textAlign": "center"}),

    html.A("Back to Home", href="/", style={"display": "block", "textAlign": "center", "marginTop": "20px", "color": "#ffffff"})
], style={"backgroundColor": "#282c34", "padding": "50px", "minHeight": "100vh"})


# Callback to update column dropdown based on selected dataset
@callback(
    Output("column-dropdown", "options"),
    Input("dataset-dropdown", "value")
)
def update_column_dropdown(selected_dataset):
    # Get columns for the selected dataset
    df = datasets[selected_dataset]
    columns = [{"label": col, "value": col} for col in df.columns if col != "timestamp"]
    return columns


# Callback to update dataset graph based on selected column
@callback(
    Output("dataset-graph", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("column-dropdown", "value")]
)
def update_dataset_graph(selected_dataset, selected_column):
    if selected_dataset and selected_column:
        df = datasets[selected_dataset]
        return {
            "data": [{
                "x": df["timestamp"],
                "y": df[selected_column],
                "type": "line",
                "name": selected_column
            }],
            "layout": {
                "title": f"{selected_dataset} - {selected_column}",
                "xaxis": {"title": "Timestamp"},
                "yaxis": {"title": selected_column},
                "template": "plotly_dark"
            }
        }
    return {}


# Callback to handle file upload
@callback(
    [Output("dataset-dropdown", "options"),
     Output("dataset-dropdown", "value")],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def upload_dataset(contents, filename):
    if contents is not None:
        # Decode the uploaded file
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            # Try to read the uploaded file as a CSV
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            # Ensure the dataset has a timestamp column
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise ValueError("Dataset must include a 'timestamp' column")

            # Add the dataset to the datasets dictionary
            datasets[filename] = df
            dataset_columns[filename] = df.columns.tolist()

            # Update dropdown options
            options = [{"label": name, "value": name} for name in datasets.keys()]
            return options, filename
        except Exception as e:
            print(f"Error processing file: {e}")
            return [{"label": name, "value": name} for name in datasets.keys()], "Dataset 1"

    return [{"label": name, "value": name} for name in datasets.keys()], "Dataset 1"
