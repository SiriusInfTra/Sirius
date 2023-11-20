FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
  python3 python3-pip python-is-python3 python3-dev python3-setuptools \
  gcc llvm libtinfo-dev zlib1g-dev build-essential libtool autoconf unzip\
  libedit-dev libxml2-dev ninja-build libboost-all-dev \
  git curl vim git-lfs wget && git lfs install

RUN wget https://cmake.org/files/v3.27/cmake-3.27.8-linux-x86_64.sh && \
  mkdir /opt/cmake && \
  sh cmake-3.27.8-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
  ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
  rm cmake-3.27.8-linux-x86_64.sh

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
