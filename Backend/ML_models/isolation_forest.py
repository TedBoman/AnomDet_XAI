import model_interface
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import numpy as np


class Isolation_Forest(model_interface.ModelInterface):

    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination='auto', random_state=None)
        return
    def run(self, df, TIME_STEPS):

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        self.model.fit(train_df)
        return

    def detect(self, detection_df):
        predictions = self.model.predict(detection_df)
        boolean_anomalies = predictions == -1
        return boolean_anomalies

