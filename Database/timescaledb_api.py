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
        self.chunk_size = 128   # The size of the chunks to split the data into when inserting into the database

    # Helper function to convert a timestamp from epoch to a datetime object
    def __add_to_timestamp(self, x: str):
        return datetime.fromtimestamp(x)

    # Helper function to insert data into the database
    def __inserter(self, query, chunk):
        try:
            retry = 0

            while retry < 5:
                conn = psycopg2.connect(self.connection_string)     # Connect to the database
                if conn:
                    break
                else:
                    time = 3
                    while time > 0:
                        print("Retrying in: {time}s")
                        sleep(1)
                        time -= 1
                retry += 1
            extras.execute_values(conn.cursor(), query, chunk)
            conn.commit()
        except Exception as error:
            print("Error: %s" % error)
            conn.rollback()
            conn.close()
        finally:
            conn.close()
    
    # Creates a hypertable called table_name with column-names columns copied from dataset
    # Also adds columns is_anomaly and injected_anomaly
    def create_table(self, table_name: str, columns: list[str]) -> None:
        length = len(columns)
        
        # The first column is of type TIMESTAMPTZ NOT NULL and the rest are VARCHAR(50)
        columns[0] = f'\"{columns[0]}\" TIMESTAMPTZ NOT NULL'
        for i in range(1, length):
            columns[i] = f'\"{columns[i]}\" VARCHAR(50)'
        columns = columns + ["is_anomaly BOOLEAN"] + ["injected_anomaly BOOLEAN"]

        for column in columns:
            print(column)

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

    # Inserts data into the table_name table. The data is a pandas DataFrame with matching types and column names to the table
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        columns = data.columns.to_list()
        cols = ', '.join([f'"{col}"' for col in columns])               # Create a string of the column names
        query_insert_data = f"INSERT INTO \"{table_name}\" ({cols}) VALUES %s"
        
        data = data.astype(str)                                         # Convert all data to strings

        first_column = data.columns[0]

        # Convert the first column to a timestamp
        data[first_column] = data[first_column].astype('int32')         
        data[first_column] = data[first_column].apply(self.__add_to_timestamp)

        tuples = [tuple(x) for x in data.to_numpy()]                    # Convert the dataframe to a list of tuples

        try:

            #length = len(tuples)
            length = self.chunk_size

            if length > self.chunk_size:                        # If the data is too large to insert at once, do multiple inserts
                num_processes = mp.cpu_count()
                pool = mp.Pool(processes=num_processes)
                
                results = []                                    # Create a list to store results from async processes
                
                print("Starting to insert!")
                inserter = self.__inserter
                print(inserter)
                print(self.create_table)
                # Insert the data in chunks                  
                for chunk in [tuples[i:i + self.chunk_size] for i in range(0, length, self.chunk_size)]:
                    result = pool.apply_async(inserter, args=(query_insert_data, chunk))
                    results.append(result)

                # Wait for all processes to finish
                pool.close()
                pool.join()


                # Check if any of the processes failed
                for result in results:
                    result.get()
            else:
                self.__inserter(query_insert_data, tuples)

        except Exception as error:
            print("Error: %s" % error)

    def insert_data_no_helper(self, table_name: str, data: pd.DataFrame):
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
    def read_data(self, table_name: str, time: datetime) -> pd.DataFrame:
        # Assuming the docker container is started, connect to the database
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f'SELECT * FROM {table_name} LIMIT 15;'
            cursor.execute(query)

            data = cursor.fetchall()
            
            df = pd.DataFrame(data)
            df.columns = [desc[0] for desc in cursor.description]

        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()
            return df

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