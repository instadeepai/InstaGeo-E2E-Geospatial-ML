# syntax = docker/dockerfile:1.2
FROM nvcr.io/nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

# Build args for user/group ID should be set to local user to avoid permission issues on volumes
ARG HOST_UID=1000
ARG HOST_GID=1000

# Avoid unnecessary writes to disk
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Set CUDA device IDs to be ordered by PCR bus IDs (see https://stackoverflow.com/a/43131539)
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install system-level dependencies required by geospatial Python libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libproj-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    curl \
    git \
    ssh \
    jq \
    wget \
    && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

# Create a working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# 1. Create a virtual environment at /app/venv
# 2. Upgrade pip in that venv
# 3. Install requirements in that venv
RUN python3 -m venv /app/venv \
    && /app/venv/bin/python -m pip install --upgrade pip \
    && /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your code into the container
COPY . /app

# Install InstaGeo (or your package) in the venv
RUN /app/venv/bin/pip install .[all]

RUN /app/venv/bin/pip uninstall -y neptune-client \
    && /app/venv/bin/pip install neptune

# Update PATH so the container defaults to using the venv’s python/pip
ENV PATH="/app/venv/bin:$PATH"

# Start an interactive shell
CMD ["bash"]
