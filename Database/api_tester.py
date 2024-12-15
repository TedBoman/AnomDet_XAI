import pandas as pd
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv
from timescaledb_api import TimescaleDBAPI

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('DATABASE_PORT'))
USER = os.getenv('POSTGRES_USER')
PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB = os.getenv('POSTGRES_DB')

# Assuming the docker container is started, connect to the database
conn_params = {
    "user": USER,
    "password": PASSWORD,
    "host": HOST,
    "port": PORT,
    "database": DB
}

api = TimescaleDBAPI(conn_params)

df = pd.read_csv("../Backend/Datasets/system-1.csv", low_memory=False)  # Read the csv file

api.create_table("system1", df.columns.to_list())                       # Create a table in the database

api.insert_data("system1", df)                                          # Insert the data into the database

api.read_data("system1", datetime.fromtimestamp(0))                     # Read the data from the database

api.drop_table("system1")                                               # Drop the table from the database