FROM ubuntu:18.04

## Python installation ##
RUN apt-get update \
    && apt-get install -y python3 \
        python3-dev \
        python3-pip \
        python3-numpy \
        pkg-config \
        git \
        wget \
        unzip \
        cmake \
        build-essential
RUN apt-get install -y \
        libsm6 \
        libxrender-dev
ARG PHP
ARG USER
ENV USER=${USER}
RUN adduser --shell /bin/bash --disabled-password --gecos "" ${USER}
ENV HOME /home/${USER}
WORKDIR /home/${USER}/
USER ${USER}
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
COPY release /home/${USER}/parking-release
WORKDIR /home/${USER}/parking-release
