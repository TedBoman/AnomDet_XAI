import psycopg2
from anom_secrets.anom_secrets import SECRET_NAME

# Assuming the docker container is started, connect to the database
CONNECTION = "postgres://Anomdet:G5anomdet@localhost:5432/mytimescaleDB"
conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor()

cursor.execute(f'DROP TABLE {SECRET_NAME};')
conn.commit()

cursor.close