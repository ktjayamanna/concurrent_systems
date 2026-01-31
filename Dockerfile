FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    tree \
    git \
    nano \
    g++ \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements file
COPY requirements.txt /code/

# Install Python dependencies
RUN pip3 install -r /code/requirements.txt

WORKDIR /code