# FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ARG PROJECT_DIR
ARG DOCKER_BUILD_DIR

ARG DEBIAN_FRONTEND=noninteractive
ARG PROXY

ENV TVM_HOME=/opt/tvm
ENV TORCH_HOME=/opt/torch
ENV TORCH_VISION_HOME=/opt/vision
ENV VLLM_HOME=/opt/vllm
ENV BOOST_HOME=/opt/boost

# -----------------------------------------------

RUN apt update && apt install -y \
  gcc gdb llvm llvm-dev libtinfo-dev zlib1g-dev build-essential libtool autoconf unzip \
  libedit-dev libxml2-dev ninja-build libgoogle-glog-dev \
  libpng-dev libjpeg-dev \
  datacenter-gpu-manager \
  git curl tmux vim neovim git-lfs wget fish htop docker.io && \
  git lfs install && \
  chsh -s /usr/bin/fish

RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
    wget https://cmake.org/files/v3.27/cmake-3.27.8-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.27.8-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    ln -s /opt/cmake/bin/ccmake /usr/local/bin/ccmake && \
    rm cmake-3.27.8-linux-x86_64.sh && \
    if [ -n "$PROXY" ]; then \
      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
    fi


SHELL ["/bin/bash", "-c"]

RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
  wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh && \
  if [ -n "$PROXY" ]; then \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
  fi && \
  bash Mambaforge-24.3.0-0-Linux-x86_64.sh -b -p /opt/mambaforge && \
  source /opt/mambaforge/bin/activate && \
  conda create -n colserve python=3.12 && \
  conda activate colserve && \
  conda install -y conda-forge::python-devtools nvitop && \
  conda init fish && \
  rm Mambaforge-24.3.0-0-Linux-x86_64.sh

# conda boost require higher version glibc, 
# so we install boost via compile from source
# (conda install -y conda-forge::boost>=1.82)


COPY ${PROJECT_DIR}/environment/requirements.txt /tmp/requirements.txt
RUN source /opt/mambaforge/bin/activate colserve && \
  pip install --upgrade pip && \
  pip install -r /tmp/requirements.txt && \
  conda install -y conda-forge::c-ares 


COPY ${PROJECT_DIR}/scripts/install_boost.sh /tmp/install_boost.sh
RUN source /opt/mambaforge/bin/activate colserve && \
  bash /tmp/install_boost.sh ${BOOST_HOME}
  

# Install TVM, PyTorch, vLLM
COPY tvm ${TVM_HOME}
COPY ${PROJECT_DIR}/scripts/build_tvm.sh /tmp/build_tvm.sh
# Ref: https://tvm.apache.org/docs/install/from_source.html
RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
    source /opt/mambaforge/bin/activate colserve && \
    rm -rf ${TVM_HOME}/.git && \
    bash /tmp/build_tvm.sh ${TVM_HOME} && \
    rm -rf /tmp/build_tvm.sh && \
    if [ -n "$PROXY" ]; then \
      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
    fi


COPY pytorch ${TORCH_HOME}
COPY torch-vision ${TORCH_VISION_HOME}
COPY ${PROJECT_DIR}/scripts/build_torch.sh /tmp/build_torch.sh
RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
    source /opt/mambaforge/bin/activate colserve && \
    rm -rf ${TORCH_HOME}/.git && \
    bash /tmp/build_torch.sh ${TORCH_HOME} ${TORCH_VISION_HOME} && \
    rm -rf /tmp/build_torch.sh && \
    if [ -n "$PROXY" ]; then \
      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
    fi
    
ARG TVM_MODEL_HOME=tvm-models
COPY ${PROJECT_DIR} /gpu-col
COPY tvm-models /gpu-col/server/models
ARG TRITON_MODEL_HOME=triton-models
COPY triton-models /gpu-col/server/triton_models

COPY ${PROJECT_DIR}/scripts/build_sirius.sh /tmp/build_sirius.sh
RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
  source /opt/mambaforge/bin/activate colserve && \
  bash /tmp/build_sirius.sh /gpu-col ${TVM_HOME} ${TORCH_HOME} ${BOOST_HOME} && \
  rm -rf /tmp/build_sirius.sh && \
  if [ -n "$PROXY" ]; then \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
  fi


COPY vllm ${VLLM_HOME}
COPY ${PROJECT_DIR}/scripts/build_vllm.sh /tmp/build_vllm.sh
RUN if [ -n "$PROXY" ]; then \
      export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY; \
    fi && \
  source /opt/mambaforge/bin/activate colserve && \
  bash /tmp/build_vllm.sh ${VLLM_HOME} && \
  rm -rf /tmp/build_vllm.sh && \
  if [ -n "$PROXY" ]; then \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
  fi

RUN apt install -y rsync

ENTRYPOINT ["/usr/bin/fish"]
