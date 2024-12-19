import pandas as pd
import numpy as np
from DBAPI.talk_to_backend import AnomalySetting
import datetime as dt

class SpikeAnomaly():

    def inject_anomaly(data, rng, magnitude):
        print("Injecting spike anomaly!")
        random_factors = rng.uniform(1, magnitude)
        return data * random_factors