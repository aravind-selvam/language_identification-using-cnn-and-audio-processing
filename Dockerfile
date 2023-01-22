FROM python:3.10-slim-bullseye

RUN apt update -y && apt install ffmpeg -y

RUN pip3 --no-cache-dir install --upgrade awscli

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]