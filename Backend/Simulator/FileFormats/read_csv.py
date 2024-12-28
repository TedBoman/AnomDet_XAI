import pandas as pd

class read_csv:
    def __init__(self, file_path):
        self.file_path = file_path

    def filetype_csv(self):
        """
        Processes a CSV file, injects anomalies, and inserts the data into the database.
        Ensures consistent anomaly injection across chunks.
        """

        full_df = pd.read_csv(self.file_path)

        return full_df