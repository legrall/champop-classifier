FROM python:3.6.8-slim as base

WORKDIR /app

RUN apt-get -y update && apt-get install -y ffmpeg sox \
   libsox-fmt-all \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install .

EXPOSE 8080

CMD [ "gunicorn", "wsgi", "--bind=0.0.0.0:8080", "--access-logfile=-"]
