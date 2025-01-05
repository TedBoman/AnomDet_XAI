from dash import Dash, dcc, html, Input, Output, State, ALL, callback, callback_context
import json
from get_handler import get_handler

def get_index_callbacks(app):
    @app.callback(
        Output("injection", "style"),
        Input("injection-check", "value")
    )
    def update_injection_panel(selected):
        if "use_injection" in selected:
            return {"display": "block"}
        return {"display": "none"}
    @app.callback(
        Output("column-dropdown", "options"),
        Input("dataset-dropdown", "value")
    )
    def update_column_dropdown(selected_dataset):
        print(selected_dataset)
        handler = get_handler()
        columns = handler.handle_get_dataset_columns(selected_dataset)
        return [{"label": col, "value": col} for col in columns]

    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-jobs-list", "children")
    )
    def toggle_active_jobs_section(children):
        # Show the active jobs section if there are active jobs
        if len(children) > 0:
            return {"display": "block", "marginTop": "30px"}
        return {"display": "none"}
        
    # Callback to remove an active job
    @app.callback(
        Output("active-jobs-list", "children"),
        Input({"type": "confirm-box", "index": ALL}, "submit_n_clicks")
    )
    def remove_active_job(n_clicks):
        handler = get_handler()
        active_jobs = handler.handle_get_running()

        ctx = callback_context

        triggered_index = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    # Callback to add and manage active jobs
    @app.callback(
        [Output("active-jobs-list", "children")],
        [Input("job-interval", "n_intervals")]
    )
    def manage_active_jobs(n_intervals):
        handler = get_handler()
        active_jobs = handler.handle_get_running()

        return [
            html.Div([
                dcc.Link(
                    dataset,
                    href=f"/{job}",
                    style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
                ),
                html.Button("Stop", id={"type": "remove-dataset-btn", "index": job}, n_clicks=0, style={
                    "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                    "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
                })
            ]) for job in active_jobs
        ]
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

    @app.callback(
            [Output("popup", "style"), Output("popup-interval", "disabled"), Output("popup", "children")],
            [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
            [
                State("dataset-dropdown", "value"),
                State("detection-model-dropdown", "value"),
                State("mode-selection", "value"),
                State("name-input", "value"),
                State("injection-method-dropdown", "value"),
                State("timestamp-input", "value"),
                State("magnitude-input", "value"),
                State("percentage-input", "value"),
                State("duration-input", "value"),
                State("column-dropdown", "value"),
                State("injection-check", "value"),
                State("popup", "style")
            ]
            )
    def start_job_handler(
                            n_clicks,
                            n_intervals,
                            selected_dataset,
                            selected_detection_model,
                            selected_mode,
                            job_name,
                            selected_injection_method,
                            timestamp,
                            magnitude,
                            percentage,
                            duration,
                            columns,
                            inj_check,
                            style
                        ):   
        handler = get_handler()
        children = "Job has started!"

        ctx = callback_context
        if not ctx.triggered:
            return style, True, children

        trigger = ctx.triggered[0]["prop_id"]
        if trigger == "start-job-btn.n_clicks" and n_clicks:
            response = handler.check_name(job_name)
            if response == "success":
                if selected_injection_method == []:
                    inj_params = None
                else: 
                    inj_params = {
                                    "anomaly_type": selected_injection_method,
                                    "timestamp": str(timestamp),
                                    "magnitude": str(magnitude),
                                    "percentage": str(percentage),
                                    "duration": str(duration),
                                    "columns": columns
                                }
                if selected_mode == "batch":
                    response = handler.handle_run_batch(selected_dataset, selected_detection_model, job_name, inj_params)
                else:
                    response = handler.handle_run_stream(selected_dataset, selected_detection_model, job_name, inj_params)
                style.update({"backgroundColor": "#4CAF50"})
            else:
                style.update({"backgroundColor": "#e74c3c"})
                children = "Job name already exists!"
            style.update({"display": "block"})
            return style, False, children
        elif trigger == "popup-interval.n_intervals":
            style.update({"display": "none"})
            return style, True, children 

        return style, True, children

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