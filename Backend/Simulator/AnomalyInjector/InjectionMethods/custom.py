import pandas as pd
import numpy as np
import datetime as dt


class CustomAnomaly():

    def inject_anomaly(self, data, magnitude):
        print("Injecting custom anomaly!")
        return data * magnitude