import psycopg2
import psycopg2.extras as extras 
import pandas as pd
import ntpath
from datetime import datetime

def execute_values(conn, arr, table, columns):
    cursor = conn.cursor()
    tuples = [tuple(x) for x in arr] 
  
    cols = ','.join(columns)

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

def main():
    file_path = "../Datasets/Electric_Production.csv"       # Path to the csv file
    df = pd.read_csv(file_path, low_memory=False)           # Read the csv file
    file_base_name = ntpath.basename(file_path)             # Get the file name and file extension
    table_name = file_base_name.split('.')[0]               # Get the file name without the file extension and use as table name

    columns = df.columns.to_list()                          # Get the columns of the csv file into a list
    first_column = columns[0]                               # Get the first column
    columns = columns[1:]                                   # Remove the first column from the list
    for i in range(len(columns)):                        # Add VARCHAR(50) to each column except first
        columns[i] = columns[i] + " VARCHAR(50)"        
    columns = ",".join(columns)                             # Join the columns with a comma

    # Assuming the docker container is started, connect to the database
    CONNECTION = "postgres://Anomdet:G5anomdet@localhost:5432/mytimescaleDB"
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # Create query for creating a relational tabel
    query_create_table = f'CREATE TABLE {table_name} ( {first_column} TIMESTAMPTZ NOT NULL, {columns});'

    # Exectute the query, creating a table in the database
    cursor.execute(query_create_table)
    conn.commit()

    # Insert data into the database
    execute_values(conn, df.to_numpy(), table_name, df.columns)                              

    # Close connection to the database
    cursor.close()

if __name__ == "__main__": 
    main()