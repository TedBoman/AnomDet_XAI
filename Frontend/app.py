import dash
from dash import dcc, html, Input, Output, State
from pages.display_data import layout as data_layout
from pages.display_data import get_local_callback
from pages.index import layout as index_layout
from callbacks import get_index_callbacks, get_display_callbacks
import os
from dotenv import load_dotenv
from get_handler import get_handler

load_dotenv()
HOST = 'Backend'
PORT = int(os.getenv('BACKEND_PORT'))
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT'))

# Dash application
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True  # Suppress errors for dynamic content

# Main layout
app.layout = html.Div([
    dcc.Store(id="store-data"),
    dcc.Location(id="url", refresh=False),  # Monitors URL changes
    html.Div(id="page-content")            # Dynamic content is updated here
])


handler = get_handler()

# Callback: Display the correct page content based on the URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("url", "search")
)
def display_page(pathname, search_string):
    if pathname == "/":  # Ensure the root URL shows the Starter Page
        return index_layout(handler)
    else:
        if handler.check_name(pathname[1:]) == "name-error":
            value = search_string.replace("?batch=", "")
            if value == "False":
                return data_layout(handler, pathname[1:], batch=False)
            return data_layout(handler, pathname[1:])
        else:
            return html.Div("404 - Page Not Found", style={"textAlign": "center", "color": "#000000"})

# Get callbacks from other pages
get_index_callbacks(app)
get_display_callbacks(app)
get_local_callback(app)


if __name__ == "__main__":
    print("Starting the Dash server...")
    app.run_server(debug=True, host="0.0.0.0", port=FRONTEND_PORT)
