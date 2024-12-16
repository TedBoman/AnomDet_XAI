import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
HOST = 'localhost'
PORT = int(os.getenv('DATABASE_PORT'))
USER = os.getenv('POSTGRES_USER')
PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB = os.getenv('POSTGRES_DB')

connection_string = f'postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}'
conn = psycopg2.connect(connection_string)
cursor = conn.cursor()

query = f'DROP TABLE system1;'
cursor.execute(query)
conn.commit()
conn.close()