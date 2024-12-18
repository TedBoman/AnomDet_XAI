from typing import List, Dict, Union

class AnomalySetting:
    def __init__(self, anomaly_type: str, timestamp: int, magnitude: int, 
                 percentage: int, columns: List[str] = None, duration: str = None):
        self.anomaly_type = anomaly_type
        self.timestamp = timestamp
        self.magnitude = magnitude
        self.percentage = percentage
        self.duration = duration
        self.columns = columns

class Message:
    def __init__(self, filepath: str, anomaly_settings: List[AnomalySetting], simulation_type,speedup: int = None, table_name: str = None):
        self.filepath = filepath
        self.anomaly_settings = anomaly_settings
        self.simulation_type = simulation_type
        self.speedup = speedup
        self.table_name = table_name

class TalkToBackend:

    def Test(self):
        filepath = './Datasets/system-1.csv'
        TestAnomaly = AnomalySetting('lowered', 90, 2, 100, ["load-5m", "load-1m"], '2m')
        TestAnomaly2 = AnomalySetting('spike', 330, 2, 100, ["load-5m", "load-1m"])

        TestMessage = Message(filepath, [TestAnomaly, TestAnomaly2], 'stream', 12)
        #Uncomment to run the simulator on test dataset with 50 rows.
        return TestMessage
        
        #return None

    def SendColumns(columns):
        print("Sending columns to backend!")
        print("WARNING! NOT IMPLEMENTED YET")

    def ReadFromBackEnd(message=None) -> Union[str, Message, "TalkToBackend.Data"]:
        print("Reading message from backend")
        print("WARNING! NOT IMPLEMENTED YET")

    def SendWarning(message, severity):
        print("Reading message from backend")
        print("WARNING! NOT IMPLEMENTED YET")

