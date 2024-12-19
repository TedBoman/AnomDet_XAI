
FROM python:3.10-slim

ARG FRONTEND_PORT

WORKDIR /app

COPY ./Frontend/ /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./Backend/backend_api.py /app/pages/

EXPOSE ${FRONTEND_PORT}

CMD ["python", "pages/app.py"]
