import sys
import pandas as pd
import numpy as np
import datetime as dt

class LoweredAnomaly():

    # In LoweredAnomaly class
    def inject_anomaly(self, data, rng, data_range, mean):
        print(f"Input data to LoweredAnomaly: {data}")
        print("Injecting lowered anomaly!")
        
        random_factors = rng.uniform(0.3, 0.4, size=len(data))
        
        if data_range == 0:
            step_values = mean * random_factors
        else:
            step_values = data_range * random_factors

        print(f"Step: {step_values}")
        
        result = np.maximum(data - step_values, 0)
        # Replace any zeros with scaled original values
        zero_mask = (result == 0)
        result[zero_mask] = data[zero_mask] * random_factors[zero_mask]
        
        return result