import pandas as pd
import random

def generate_fake_data():
    # 100 dakikalık sahte performans verisi oluştur
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='T')
    load_15m = [random.uniform(0, 10) for _ in range(100)]  # 15 dakikalık yük
    return pd.DataFrame({'timestamp': timestamps, 'load_15m': load_15m})
