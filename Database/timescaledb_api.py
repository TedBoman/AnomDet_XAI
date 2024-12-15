from db_interface import DBInterface
import pandas as pd
from datetime import datetime
import psycopg2
import psycopg2.extras as extras
import multiprocessing as mp

class TimescaleDBAPI(DBInterface):
    # Initialize the connection string that psycopg2 uses to connect to the database
    def __init__(self, conn_params: dict):
        user = conn_params["user"]
        password = conn_params["password"]
        host = conn_params["host"]
        port = conn_params["port"]
        database = conn_params["database"]
    
        self.connection_string = f'postgres://{user}:{password}@{host}:{port}/{database}'
        self.chunk_size = 32   # The size of the chunks to split the data into when inserting into the database
    
    # Creates a hypertable called table_name with column-names columns copied from dataset
    # Also adds columns is_anomaly and injected_anomaly
    def create_table(self, table_name: str, columns: list[str]):
        length = len(columns)
        
        # The first column is of type TIMESTAMPTZ NOT NULL and the rest are VARCHAR(50)
        columns[0] = f'\"{columns[0]}\" TIMESTAMPTZ NOT NULL'
        for i in range(1, length):
            columns[i] = f'\"{columns[i]}\" VARCHAR(50)'
        columns = columns + ["is_anomaly BOOLEAN"] + ["injected_anomaly BOOLEAN"]

        try: 
            conn = psycopg2.connect(self.connection_string)                         # Connect to the database
            cursor = conn.cursor()
            
            query_create_table = f'CREATE TABLE {table_name} ({",".join(columns)});'# Create query for creating a relational tabel
            
            cursor.execute(query_create_table)                                      # Exectute the query, creating a table in the database
            conn.commit()
        except Exception as error:
            print("Error: %s" % error)
            conn.close() 
        finally:
            conn.close()

    # Inserts data into the table_name table. The data is a pandas DataFrame with matching types and column names to the table
    def insert_data(self, table_name: str, data: pd.DataFrame):
        cols = ', '.join([f'"{col}"' for col in data.columns.to_list()])
        query = f"INSERT INTO \"{table_name}\" ({cols}) VALUES %s"
        length = len(data.columns)

        data = data.astype(str)                                         # Convert all data to strings

        first_column = data.columns[0]

        # Convert the first column to a timestamp
        data[first_column] = data[first_column].astype('int32')         
        data[first_column] = data[first_column].apply(self.__add_to_timestamp)

        tuples = [tuple(x) for x in data.to_numpy()]                    # Convert the dataframe to a list of tuples

        try:
            conn = psycopg2.connect(self.connection_string)     # Connect to the database

            if length > self.chunk_size:                        # If the data is too large to insert at once, do multiple inserts
                num_processes = mp.cpu_count()
                pool = mp.Pool(processes=num_processes)
                
                results = []                                    # Create a list to store results from async processes

                # Insert the data in chunks                  
                for chunk in [tuples[i:i + self.chunk_size] for i in range(0, length, self.chunk_size)]:
                    result = pool.apply_async(self.__inserter, args=(conn, query, chunk))
                    results.append(result)

                # Wait for all processes to finish
                pool.close()
                pool.join()

                # Check if any of the processes failed
                for result in results:
                    result.get()
            else:
                self.__inserter(conn, query, tuples)

        except Exception as error:
            print("Error: %s" % error)
            conn.rollback()
            conn.close()
        finally:
            conn.close()
    
    # Reads each row of data in the table table_name that has a timestamp greater than or equal to time
    def read_data(self, table_name: str, time: datetime):
        # Assuming the docker container is started, connect to the database
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            query = f'SELECT * FROM {table_name} LIMIT 15;'
            cursor.execute(query)
            for row in cursor.fetchall():
                print(row)
        except Exception as error:
            print("Error: %s" % error)
            conn.close()
        finally:
            conn.close()

    # Deletes the table_name table along with all its data
    def drop_table(self, table_name: str):
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

    # Helper function to insert data into the database
    def __inserter(self, conn, query, chunk):
        extras.execute_values(conn.cursor(), query, chunk)
        conn.commit()

    def __add_to_timestamp(self, x: str):
        return datetime.fromtimestamp(x)