# syntax = docker/dockerfile:1.2

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.6.2
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ENV PYTHONDONTWRITEBYTECODE=true \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=true \
    UV_LINK_MODE=copy \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

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
    parallel \
    openssl \
    && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    GDAL_DATA=/usr/share/gdal

# Create non-root user
RUN groupadd -g 1001 instageo && \
    useradd -u 1001 -g instageo --shell /bin/bash --create-home instageo && \
    mkdir -p /app && chown -R instageo:instageo /app

FROM base AS with-uv

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

FROM with-uv AS with-deps

# Redeclare ARG for this stage
ARG PYTHON_VERSION=3.11

# Copy requirements files first for better caching
COPY uv.lock /app/
COPY pyproject.toml /app/

WORKDIR /app

# Create virtual environment as root, then fix ownership
RUN uv venv /app/venv --python ${PYTHON_VERSION} && \
    chown -R instageo:instageo /app/venv

# Add venv to PATH
ENV PATH="/app/venv/bin:$PATH"

FROM with-deps AS with-code

COPY . /app/instageo
RUN chown -R instageo:instageo /app/instageo

WORKDIR /app/instageo

# Cache uv downloads for faster builds
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --frozen

# Switch to non-root user
USER instageo

# Start an interactive shell
CMD ["bash"]
