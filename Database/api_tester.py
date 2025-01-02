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

df = pd.read_csv("../Backend/Datasets/test_system.csv", low_memory=False)  # Read the csv file

api.create_table("test", df.columns.to_list())                       # Create a table in the database
columns = api.get_columns("test")                                    # Get the columns of the table

print(columns)

df["is_anomaly"] = False
df["injected_anomaly"] = False

start_time = time.time()
api.insert_data("test", df)                                          # Insert the data into the database
print(f"Time to insert data: {time.time() - start_time}")

df = api.read_data("test", datetime.fromtimestamp(0))                # Read the data from the database
print(df)

api.drop_table("test")                                               # Drop the table from the database