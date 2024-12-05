import psycopg2
import psycopg2.extras as extras 
import pandas as pd
import ntpath
from anom_secrets.anom_secrets import SECRET_PATH
import time
from datetime import datetime

def add_to_timestamp(x: str):
    return datetime.fromtimestamp(x)

def execute_values(conn, df, table, columns):
    cursor = conn.cursor()
    first_column = columns[0].replace("\"", '')             # Remove the double quotes from the first column name
    df = df.astype('str')
    print(type(df[first_column].iloc[0]))
    df[first_column] = df[first_column].astype('int32').apply(add_to_timestamp)
    print(df[first_column].iloc[0])
    tuples = [tuple(x) for x in df.to_numpy()]              # Convert the dataframe to a list of tuples
    cols = ','.join(columns)
    print(tuples[0])

    start_time = time.time()

    # SQL query to execute 
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols) # Creates the query: INSERT INTO table_name(column1, column2, ...) VALUES %s
                                                            # %s is a placeholder for the tuple values inserted by extras.execute_values()
    try: 
        extras.execute_values(cursor, query, tuples) 
        conn.commit() 
    except (Exception, psycopg2.DatabaseError) as error: 
        print("Error: %s" % error) 
        conn.rollback() 
        cursor.close() 
    print("execute_values() done") 
    print("Inserting values took %s seconds" % (time.time() - start_time))

def main():
    file_path = SECRET_PATH                                           # Path to the csv file
    df = pd.read_csv(file_path, low_memory=False)                     # Read the csv file
    file_base_name = ntpath.basename(file_path)                       # Get the file name and file extension
    table_name = file_base_name.split('.')[0]                         # Get the file name without the file extension and use as table name

    columnn_names = df.columns.to_list()                              # Get the columns of the csv file into a list
    columns_with_types = columnn_names.copy()
    for i in range(len(columnn_names)):                               # Add VARCHAR(50) to each column except first
        columnn_names[i] = f'\"{columnn_names[i]}\"'
        columns_with_types[i] = f'{columnn_names[i]} VARCHAR(50)'    
    columns_with_types[0] = f'{columnn_names[0]} TIMESTAMPTZ NOT NULL'

    # Assuming the docker container is started, connect to the database
    CONNECTION = "postgres://Anomdet:G5anomdet@localhost:5432/mytimescaleDB"
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # Create query for creating a relational tabel
    query_create_table = f'CREATE TABLE {table_name} ({",".join(columns_with_types)});'

    # Exectute the query, creating a table in the database
    cursor.execute(query_create_table)
    conn.commit()

    # Insert data into the database
    execute_values(conn, df, table_name, columnn_names)                              

    # Close connection to the database
    cursor.close()

if __name__ == "__main__": 
    main()