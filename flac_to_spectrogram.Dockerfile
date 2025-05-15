FROM docker.io/python:3.12.7-slim-bookworm

RUN apt update
RUN apt install -y ffmpeg

WORKDIR /usr/src/app

COPY flac_to_spectrogram.requirements ./
RUN pip install --no-cache-dir -r flac_to_spectrogram.requirements

COPY flac_to_spectrogram.py .
RUN chmod +x flac_to_spectrogram.py

RUN rm -rf /var/lib/apt/lists/*
RUN rm ./flac_to_spectrogram.requirements
