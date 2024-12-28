import pandas as pd
import psycopg2
import os
from datetime import datetime
import time
from dotenv import load_dotenv
from timescaledb_api import TimescaleDBAPI

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('DATABASE_PORT'))
USER = os.getenv('DATABASE_USER')
PASSWORD = os.getenv('DATABASE_PASSWORD')
NAME = os.getenv('DATABASE_NAME')

# Assuming the docker container is started, connect to the database
conn_params = {
    "user": USER,
    "password": PASSWORD,
    "host": HOST,
    "port": PORT,
    "database": NAME
}


api = TimescaleDBAPI(conn_params)

df = pd.read_csv("../Backend/Datasets/system-1.csv", low_memory=False)  # Read the csv file

api.create_table("system1", df.columns.to_list())                       # Create a table in the database
columns = api.get_columns("system1")                                    # Get the columns of the table

print(columns)

df["is_anomaly"] = False
df["injected_anomaly"] = False

start_time = time.time()
api.insert_data("system1", df)                                          # Insert the data into the database
print(f"Time to insert data: {time.time() - start_time}")

api.read_data("system1", datetime.fromtimestamp(0))                     # Read the data from the database

api.drop_table("system1")                                               # Drop the table from the database