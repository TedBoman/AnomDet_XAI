
FROM python:3.10-slim

ARG FRONTEND_PORT

WORKDIR /app

COPY ./Frontend/ /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./Backend/api/api.py /app/

EXPOSE ${FRONTEND_PORT}

CMD ["python", "-u", "app.py"]


