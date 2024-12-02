import dash
from dash import dcc, html, Input, Output
from load_data import layout as load_data_layout
from stream_data import layout as stream_data_layout, register_callbacks
from starter_page import layout as starter_page_layout
from anomaly_detection import layout as anomaly_detection_layout
# Dash uygulaması
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True  # Dinamik içerikler için hataları bastırır

# Ana layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),  # URL değişimlerini izler
    html.Div(id="page-content")            # Dinamik içerik burada değişir
])

# Callback: URL'ye göre doğru sayfa içeriğini göster
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/load-data":
        return load_data_layout
    elif pathname == "/stream-data":
        return stream_data_layout
    elif pathname == "/anomaly-detection":
        return anomaly_detection_layout
    else:
        return starter_page_layout

# Stream data için callback'leri kaydet
register_callbacks(app)

if __name__ == "__main__":
    print("Starting the Dash server...")
    app.run_server(debug=True)
