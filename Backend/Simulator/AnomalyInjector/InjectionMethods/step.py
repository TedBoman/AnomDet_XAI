import pandas as pd
import numpy as np
from DBAPI.talk_to_backend import AnomalySetting
import datetime as dt

class StepAnomaly():

    def inject_anomaly(data, mean, magnitude):
        print("Injecting step anomaly!")
        step_value = mean * magnitude
        return data + step_value