from dash import Dash, dcc, html, Input, Output, State, ALL, callback, callback_context
from pages.display_data import graphs


def get_index_callbacks(app):
    @app.callback(
        Output("injection", "style"),
        Input("injection-check", "value")
    )
    def update_injection_panel(selected):
        if "use_injection" in selected:
            return {"display": "block"}
        return {"display": "none"}

    """
    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-datasets-list", "children")
    )
    def toggle_active_jobs_section(children):
        # Visa sektionen om det finns några aktiva jobb, annars dölj den
        if children:
            return {"display": "block", "marginTop": "30px"}
        return {"display": "none"}

    # Callback to add and manage active datasets
    @app.callback(
        Output("active-datasets-list", "children"),
        [Input("start-job-btn", "n_clicks"),
        Input({"type": "remove-dataset-btn", "index": ALL}, "n_clicks")],
        [State("dataset-dropdown", "value")]
    )
    def manage_active_datasets(add_clicks, remove_clicks, selected_dataset):
        global active_datasets
        ctx = callback_context

        if not ctx.triggered:
            return [
                html.Div([
                    dcc.Link(
                        dataset,
                        href=f"/stream-data",
                        style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
                    ),
                    html.Button("Stop", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                        "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                        "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
                    })
                ]) for dataset in active_datasets
            ]

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if "start-job-btn" in triggered_id and selected_dataset:
            if selected_dataset not in active_datasets:
                active_datasets.append(selected_dataset)

        elif "remove-dataset-btn" in triggered_id:
            triggered_index = eval(triggered_id)["index"]
            active_datasets = [dataset for dataset in active_datasets if dataset != triggered_index]

        return [
            html.Div([
                dcc.Link(
                    dataset,
                    href=f"/stream-data",
                    style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
                ),
                html.Button("Stop", id={"type": "remove-dataset-btn", "index": dataset}, n_clicks=0, style={
                    "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                    "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
                })
            ]) for dataset in active_datasets
        ]

    """
    """
    @app.callback(
        [Output("popup", "style"), Output("popup-interval", "disabled")],
        [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
        [State("popup", "style")]
    )
    def handle_popup(n_clicks, n_intervals, style):
        ctx = callback_context
        if not ctx.triggered:
            return style, True

        trigger = ctx.triggered[0]["prop_id"]
        if trigger == "start-job-btn.n_clicks" and n_clicks:
            style.update({"display": "block"})
            return style, False 
        elif trigger == "popup-interval.n_intervals":
            style.update({"display": "none"})
            return style, True 

        return style, True
    """

    @app.callback(
            Output("starter-feedback", "children"),
            State("dataset-dropdown", "value"),
            State("detection-model-dropdown", "value"),
            State("mode-selection", "value"),
            State("injection-method-dropdown", "value"),
            State("date-picker-range", "start_date"),
            State("date-picker-range", "end_date"),
            Input("start-job-btn", "n_clicks")
            )
    def start_job(**kwargs):   
        print("test")

def get_display_callbacks(app):
    """
    # Update the store with selected dataset and columns based on URL
    @app.callback(
        Output("store-data", "data"),
        Input("url", "pathname")
    )
    def store_data(pathname):
        # Store the dataset name and columns selected from the URL
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
        # Update available columns based on the stored dataset
        if store_data and "dataset_name" in store_data:
            dataset_name = store_data["dataset_name"]
            columns = datasets[dataset_name].columns.tolist()
            columns.remove("timestamp")
            return [{"label": col, "value": col} for col in columns]
        return []
    """

    #Iterates through the selected columns and displays the corresponding graphs
    @app.callback(
        [Output("selected-graphs", "children"),
         Output("anomaly-log", "children")],
        [Input("stream-interval", "n_intervals"),
         Input("store-data", "data"),
         Input("column-selector", "value")]
    )
    def update_graphs(n_intervals, store_data, selected_columns):
        #im not really sure if this is going to be used later on, but i kept it here for now
        if not store_data or "dataset_name" not in store_data:
            return html.Div("Dataset not found.", style={"color": "#ffffff", "textAlign": "center"}), []
        
        filtered_graphs = [graphs[int(col)] for col in selected_columns]

        anomaly_log = []  # Initialize anomaly log
        for col in selected_columns:
            #Dont really know what to really append in the anomaly log but added this as an example
            anomaly_log.append(f"Anomaly detected in column {col}")

        return filtered_graphs, html.Ul([html.Li(log) for log in anomaly_log])


        '''
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
        return graphs, html.Ul([html.Li(log) for log in anomaly_log[-10:]])'''