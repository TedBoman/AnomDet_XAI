import psycopg2
from anom_secrets.anom_secrets import SECRET_NAME

# Assuming the docker container is started, connect to the database
CONNECTION = "postgres://Anomdet:G5anomdet@localhost:5432/mytimescaleDB"
conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor()

query = f'SELECT * FROM {SECRET_NAME} LIMIT 15;'
cursor.execute(query)
for row in cursor.fetchall():
    print(row)
cursor.close()