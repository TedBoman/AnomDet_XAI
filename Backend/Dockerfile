FROM python:3.10-slim

ARG BACKEND_PORT

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${BACKEND_PORT}

COPY . .

ENTRYPOINT ["python", "-u", "engine.py"]
