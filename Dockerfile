FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN sh -c 'apt-get update && apt-get install -y ffmpeg git'

RUN pip install -r requirements.txt

RUN python -c 'import whisper; whisper.load_model("base");'