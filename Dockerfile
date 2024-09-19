#
# Dockerfile for 
# Score-Based Point Cloud Denoising (ICCV'21)
# https://github.com/luost26/score-denoise
#

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /app/

# APT packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    nano \
    git \
    locales \
    ca-certificates \
    wget \
    g++ \
    cmake \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    build-essential \
    libeigen3-dev \
    libcgal-dev \
    libgflags-dev \
    libatlas-base-dev

# PointFilter deps
RUN pip install --upgrade setuptools && pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install numpy matplotlib scipy scikit-learn plyfile pandas tensorboard torchsummary fvcore iopath torch-cluster point-cloud-utils \
    && FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# PointFilter
RUN git clone -b new_feat https://github.com/alancneves/score-denoise score-denoise
RUN ln -s $(which python3) /usr/bin/python