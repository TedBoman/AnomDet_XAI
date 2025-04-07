FROM python:3.10-slim

ARG BACKEND_PORT

COPY ./Backend/OpenXAI/ ./OpenXAI 
COPY ./Backend/requirements.txt .

RUN pip install -r requirements.txt

EXPOSE ${BACKEND_PORT}

COPY ./Backend/ .
COPY ./Database/db_interface.py .
COPY ./Database/timescaledb_api.py .
COPY ./Docker/.env .
COPY ./Docker/.env ./api/

ENTRYPOINT ["python", "-u", "engine.py"]
