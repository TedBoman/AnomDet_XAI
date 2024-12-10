from typing import List, Dict, Union

class AnomalySetting:
    def __init__(self, anomaly_type: str, timestamp: int, magnitude: int, 
                 percentage: int, duration: str, columns: List[str]):
        self.anomaly_type = anomaly_type
        self.timestamp = timestamp
        self.magnitude = magnitude
        self.percentage = percentage
        self.duration = duration
        self.columns = columns

class Message:
    def __init__(self, filepath: str, anomaly_settings: List[AnomalySetting], speedup: int):
        self.filepath = filepath
        self.anomaly_settings = anomaly_settings
        self.speedup = speedup

class TalkToBackend:

    def Test(self):
        filepath = './Datasets/test_system.csv'
        TestAnomaly = AnomalySetting('lowered', 630, 2, 100, '2m', ["load-5m", "load-1m"])
        TestMessage = Message(filepath, [TestAnomaly], 0)
        return TestMessage

    def SendColumns(columns):
        print("Sending columns to backend!")
        print("WARNING! NOT IMPLEMENTED YET")

    def ReadFromBackEnd(message=None) -> Union[str, Message, "TalkToBackend.Data"]:
        print("Reading message from backend")
        print("WARNING! NOT IMPLEMENTED YET")

    def SendWarning(message, severity):
        print("Reading message from backend")
        print("WARNING! NOT IMPLEMENTED YET")

