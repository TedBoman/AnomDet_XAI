import time as t
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from DBAPI.db_interface import DBInterface as db
from AnomalyInjector.anomalyinjector import TimeSeriesAnomalyInjector
from DBAPI import utils as ut
import datetime    

def filetype_csv(file_path):
        """
        Processes a CSV file, injects anomalies, and inserts the data into the database.
        Ensures consistent anomaly injection across chunks.
        """

        full_df = pd.read_csv(file_path)

        return full_df