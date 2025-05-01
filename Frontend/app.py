# app.py 
import dash
from dash import dcc, html, Input, Output, State
import os
from dotenv import load_dotenv

from pages.index import layout as index_layout
from pages.job_page import layout as job_layout
from callbacks import get_index_callbacks 
from job_page_callbacks import register_job_page_callbacks

from get_handler import get_handler

load_dotenv()
HOST = 'Backend' 
PORT = int(os.getenv('BACKEND_PORT'))
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT'))

# Dash application
# Consider adding themes or external stylesheets if desired
# import dash_bootstrap_components as dbc
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, suppress_callback_exceptions = True)

# Main layout
app.layout = html.Div([
    dcc.Store(id="store-data"), 
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

handler = get_handler()

# Callback: Display the correct page content based on the URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    print(f"Routing: Pathname received: {pathname}") # Debugging line
    if pathname == "/":
        print("Routing to index page")
        return index_layout(handler)
    elif pathname and pathname.startswith("/job/"):
        # Extract job name from path like "/job/my_job_name"
        job_name = pathname.split("/job/", 1)[1]
        if not job_name: # Handle case like "/job/"
             print("Routing: Invalid job path.")
             return html.Div("Invalid Job Path", style={"textAlign": "center", "color": "orange"})
        print(f"Routing to job page for: {job_name}")
        # --- Call the layout function from job_page.py ---
        try:
            # Pass the handler instance and job_name to the job page layout
            return job_layout(handler, job_name)
        except Exception as e:
            print(f"Error generating job layout for {job_name}: {e}")
            return html.Div(f"Error loading layout for job '{job_name}'.", style={"textAlign": "center", "color": "red"})
    else:
        print(f"Routing: Path '{pathname}' not found.")
        # Return a more user-friendly 404 page
        return html.Div([
                html.H1("404 - Page Not Found", style={'color': '#E0E0E0'}),
                html.P(f"The requested path '{pathname}' was not recognized.", style={'color': '#C0C0C0'}),
                dcc.Link("Go back to Home Page", href="/", style={'color': '#7FDBFF'})
            ], style={"textAlign": "center", "padding": "50px", 'backgroundColor': '#104E78'})

get_index_callbacks(app)

register_job_page_callbacks(app)

if __name__ == "__main__":
    print(f"Starting the Dash server on http://0.0.0.0:{FRONTEND_PORT}")
    # Set debug=False for production if needed, True is useful for development
    app.run(debug=True, host="0.0.0.0", port=FRONTEND_PORT)