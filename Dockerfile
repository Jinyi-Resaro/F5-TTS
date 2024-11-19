FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN apt-get update && \
    apt-get install -y \
    wget curl man git less openssl libssl-dev \
    unzip unar build-essential aria2 tmux vim \
    openssh-server sox libsox-fmt-all libsox-fmt-mp3 \
    libsndfile1-dev ffmpeg procps && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /workspace/F5-TTS

ENV SHELL=/bin/bash