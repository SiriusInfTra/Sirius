# FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
  gcc llvm llvm-dev libtinfo-dev zlib1g-dev build-essential libtool autoconf unzip \
  libedit-dev libxml2-dev ninja-build libgoogle-glog-dev \
  libpng-dev libjpeg-dev \
  datacenter-gpu-manager \
  git curl vim neovim git-lfs wget fish htop && \
  git lfs install && \
  chsh -s /usr/bin/fish

RUN wget https://cmake.org/files/v3.27/cmake-3.27.8-linux-x86_64.sh && \
  mkdir /opt/cmake && \
  sh cmake-3.27.8-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
  ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
  ln -s /opt/cmake/bin/ccmake /usr/local/bin/ccmake && \
  rm cmake-3.27.8-linux-x86_64.sh


SHELL ["/bin/bash", "-c"]

RUN wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh && \
  bash Mambaforge-24.3.0-0-Linux-x86_64.sh -b -p /opt/mambaforge && \
  source /opt/mambaforge/bin/activate && \
  conda create -n colserve python=3.10 && \
  conda activate colserve && \
  conda install -y conda-forge::python-devtools && \
  conda install -y conda-forge::boost>=1.82 && \
  conda init fish && \
  rm Mambaforge-24.3.0-0-Linux-x86_64.sh

COPY requirements.txt /tmp/requirements.txt
RUN source /opt/mambaforge/bin/activate colserve && \
  pip install --upgrade pip && \
  pip install -r /tmp/requirements.txt

ENTRYPOINT ["/usr/bin/fish"]
