FROM python:3.10-slim

ARG BACKEND_PORT

COPY ./Backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${BACKEND_PORT}

COPY ./Backend/ .
COPY ./Database/db_interface.py .
COPY ./Database/timescaledb_api.py .

ENTRYPOINT ["python", "-u", "engine.py"]
