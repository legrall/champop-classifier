FROM python:3.6.8-slim as base

WORKDIR /app

COPY . .
RUN pip install .

EXPOSE 8080

CMD [ "gunicorn", "wsgi", "--bind=0.0.0.0:8080", "--access-logfile=-"]
