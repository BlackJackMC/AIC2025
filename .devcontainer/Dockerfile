FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl htop build-essential unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash


