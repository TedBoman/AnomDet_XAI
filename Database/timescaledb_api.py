import numpy as np
from db_interface import DBInterface
import pandas as pd
from datetime import datetime
import psycopg2
import psycopg2.extras as extras
from psycopg2.extras import execute_values
import multiprocessing as mp
from time import sleep

class TimescaleDBAPI(DBInterface):
    # Initialize the connection string that psycopg2 uses to connect to the database
    def __init__(self, conn_params: dict):
        user = conn_params["user"]
        password = conn_params["password"]
        host = conn_params["host"]
        port = conn_params["port"]
        database = conn_params["database"]
    
        self.connection_string = f'postgres://{user}:{password}@{host}:{port}/{database}'
    
    # Creates a hypertable called table_name with column-names columns copied from dataset
    # Also adds columns is_anomaly and injected_anomaly
    def create_table(self, table_name: str, columns: list[str]) -> None:
        length = len(columns)
        
        # The first column is of type TIMESTAMPTZ NOT NULL and the rest are VARCHAR(50)
        columns[0] = f'\"{columns[0]}\" TIMESTAMPTZ NOT NULL'
        for i in range(1, length):
            columns[i] = f'\"{columns[i]}\" TEXT'
        columns = columns + ["is_anomaly BOOLEAN"] + ["injected_anomaly BOOLEAN"]

        try: 
            conn = psycopg2.connect(self.connection_string)                         # Connect to the database
            cursor = conn.cursor()
            
            query_create_table = f'CREATE TABLE "{table_name}" ({",".join(columns)});'
            cursor.execute(query_create_table)
        
            # Make the table a hypertable partitioned by timestamp
            query_create_hypertable = f'SELECT create_hypertable(\'{table_name}\', \'timestamp\');'
            cursor.execute(query_create_hypertable)

            conn.commit()
                
        except Exception as error:
            print("Error: %s" % error)
            conn.close() 
            return None
        finally:
            conn.close()

    def insert_data(self, table_name: str, data: pd.DataFrame):
        """
        Inserts data into the specified table and sets the "injected_anomaly" column.

        Args:
            table_name: The name of the table.
            data: The DataFrame containing the data to insert.
            isAnomaly: A boolean indicating whether the data has been injected with an anomaly.
        """
        conn = psycopg2.connect(self.connection_string) # Connect to the database
        cursor = conn.cursor()
        with conn.cursor() as cur:
            # Add "injected_anomaly" to the columns
            columns = ', '.join([f'"{col}"' for col in data.columns])  
            query = f"INSERT INTO \"{table_name}\" ({columns}) VALUES %s"

            try:
                # Convert DataFrame to list of tuples, with type conversion and anomaly flag
                values = [tuple(
                    float(x) if isinstance(x, (np.float64, np.float32)) else x
                    for x in row
                ) for row in data.values]
                execute_values(cur, query, values)
                conn.commit()
            except Exception as e:
                conn.rollback()
    
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    def read_data(self, time: datetime, table_name: str) -> pd.DataFrame:
        # Assuming the docker container is started, connect to the database
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f'SELECT * FROM {table_name} WHERE timestamp >= \'{time}\' ORDER BY timestamp ASC;'
            cursor.execute(query)

            data = cursor.fetchall()
            conn.close()
            
            df = pd.DataFrame(data)
            df.columns = [desc[0] for desc in cursor.description]

            cols = df.columns.tolist()

            cols.remove("is_anomaly")
            cols.remove("injected_anomaly")
            cols.remove("timestamp")
            
            for col in cols:
                df[col] = df[col].astype(float)

            return df
        except Exception as error:
            print("Error: %s" % error)
            conn.close()
            

    # Deletes the table_name table along with all its data
    def drop_table(self, table_name: str) -> None:
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(f'DROP TABLE {table_name};')
            conn.commit()
        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()

    # Checks if the table_name table exists in the database
    def table_exists(self, table_name: str) -> bool:
        result = []
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{table_name}'"

            cursor.execute(query)
            result = cursor.fetchall()
        except Exception as error:
            print("Error: %s" % error)
            #conn.close()
        finally:
            #conn.close()
            if len(result) > 0:
                return True
            else:
                return False

    # Returns a list of all columns in the table_name table
    def get_columns(self, table_name: str) -> list[str]:
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"

            cursor.execute(query)
            result = cursor.fetchall()

        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()
            columns = [x[0] for x in result]
            columns.remove("is_anomaly")
            columns.remove("injected_anomaly")

            return columns

    # Updates rows of the table that have an anomaly detected
    def update_anomalies(self, table_name: str, anomalies) -> None:
    
        try: 
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            queries = []

            for anomaly in anomalies:
                queries.append(f"UPDATE {table_name} SET is_anomaly = TRUE WHERE timestamp = {anomaly};")


            cursor.execute("".join(queries))
            conn.commit()

        except Exception as e:
            conn.rollback()
            conn.close()

        finally:
            conn.close()