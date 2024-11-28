import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import random

# Dash uygulamasını başlat
app = dash.Dash(__name__)
app.title = "System Performance Dashboard"

# Sahte veri üretme fonksiyonu
def generate_fake_data():
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='min')  # 500 dakika
    data = {
        'timestamp': timestamps,
        'load_15m': [random.uniform(0, 10) for _ in range(500)],
        'cpu_usage': [random.uniform(20, 90) for _ in range(500)],
        'memory_usage': [random.uniform(30, 80) for _ in range(500)]
    }
    return pd.DataFrame(data)

# Veri
df = generate_fake_data()

# Layout (Kullanıcı arayüzü)
app.layout = html.Div([
    html.H1("System Performance Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown (Metriği seçmek için)
    dcc.Dropdown(
        id='metric-selector',
        options=[
            {'label': '15m Load', 'value': 'load_15m'},
            {'label': 'CPU Usage', 'value': 'cpu_usage'},
            {'label': 'Memory Usage', 'value': 'memory_usage'}
        ],
        value='load_15m',
        style={'width': '50%', 'margin': '0 auto'}
    ),
    
    # Zaman aralığı seçimi
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['timestamp'].min(),
        end_date=df['timestamp'].max(),
        style={'margin': '20px'}
    ),
    
    # Grafik
    dcc.Graph(id='line-chart'),

    # Otomatik veri yenileme
    dcc.Interval(
        id='interval-update',
        interval=60 * 1000,  # 60 saniyede bir güncelle
        n_intervals=0
    )
])

# Callback: Grafiği Güncelle
@app.callback(
    Output('line-chart', 'figure'),
    [Input('metric-selector', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('interval-update', 'n_intervals')]
)
def update_chart(selected_metric, start_date, end_date, n_intervals):
    # Veriyi filtrele
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    # Grafik oluştur
    figure = go.Figure([
        # Ana veri
        go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df[selected_metric],
            mode='lines+markers',
            name=selected_metric,
            line=dict(color='blue'),
        ),
        # Anomaliler
        go.Scatter(
            x=filtered_df[filtered_df[selected_metric] > 8]['timestamp'],
            y=filtered_df[filtered_df[selected_metric] > 8][selected_metric],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomalies'
        )
    ])
    figure.update_layout(
        title=f"{selected_metric} Over Time",
        xaxis_title="Timestamp",
        yaxis_title=selected_metric,
        template="plotly_dark"
    )
    return figure

# Uygulamayı çalıştır
if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8050)

