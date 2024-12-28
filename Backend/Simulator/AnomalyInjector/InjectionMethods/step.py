import pandas as pd
import numpy as np
import datetime as dt

class StepAnomaly():

    def inject_anomaly(self, data, mean, magnitude):
        print("Injecting step anomaly!")
        step_value = mean * magnitude
        return data + step_value