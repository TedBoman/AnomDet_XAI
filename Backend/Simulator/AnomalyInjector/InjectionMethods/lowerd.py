import pandas as pd
import numpy as np
from DBAPI.talk_to_backend import AnomalySetting
import datetime as dt

class LoweredAnomaly():

    def inject_anomaly(data, rng , data_range, mean):
        print("Injecting lowerd anomaly!")
        # Handle single row case
        if data_range == 0:
            # Use mean as a reference point if data_range is zero
            random_factors = rng.uniform(0.3, 0.4)
            step_value = -mean * random_factors
        else:
            random_factors = rng.uniform(0.3, 0.4)
            step_value = -data_range * random_factors

        print(f"Step: {step_value} = -datarange: -{data_range} * random: {random_factors} = {-data_range * random_factors}")
        print(f"OLD: {data}. NEW: {np.maximum(data + step_value, 0)}")
        print(f"return: {np.maximum(data + step_value, 0)}")

        return np.maximum(data + step_value, 0)