import pandas as pd
import numpy as np
from DBAPI.talk_to_backend import AnomalySetting
import datetime as dt


class CustomAnomaly():

    def inject_anomaly(data, magnitude):
        print("Injecting custom anomaly!")
        return data * magnitude