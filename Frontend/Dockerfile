
FROM python:3.10-slim

ARG FRONTEND_PORT

WORKDIR /app

COPY ./Frontend/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./Frontend/ /app/
COPY ./Backend/backend_api.py /app/

EXPOSE ${FRONTEND_PORT}

CMD ["python", "pages/app.py"]
