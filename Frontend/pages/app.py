import dash
from dash import dcc, html, Input, Output
from load_data import layout as load_data_layout
from stream_data import layout as stream_data_layout, register_callbacks
from starter_page import layout as starter_page_layout
#from anomaly_detection import layout as anomaly_detection_layout

# Dash application
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True  # Suppress errors for dynamic content

# Main layout
app.layout = html.Div([
    dcc.Store(id="store-data"),
    dcc.Location(id="url", refresh=False),  # Monitors URL changes
    html.Div(id="page-content")            # Dynamic content is updated here
])

# Callback: Display the correct page content based on the URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/load-data":
        return load_data_layout
    elif pathname == "/stream-data":
        return stream_data_layout
   # elif pathname == "/anomaly-detection":
    #    return anomaly_detection_layout
    elif pathname == "/":  # Ensure the root URL shows the Starter Page
        return starter_page_layout
    else:
        return html.Div("404 - Page Not Found", style={"textAlign": "center", "color": "#ffffff"})

# Register callbacks for stream data
register_callbacks(app)

if __name__ == "__main__":
    print("Starting the Dash server...")
    app.run_server(debug=True, host="0.0.0.0", port=8050)
