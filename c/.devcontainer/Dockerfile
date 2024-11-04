FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git \
    coreutils sudo curl wget python3 python3-pip python3-dev build-essential zip unzip && \
    rm -rf /var/lib/apt/lists/*

# Install various Python libs, insert yours here
RUN pip3 install notebook pandas sshtunnel matplotlib cupy-cuda12x numba pycuda torch

