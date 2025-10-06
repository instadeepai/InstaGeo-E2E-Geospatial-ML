# syntax=docker/dockerfile:1.2

ARG PYTHON_VERSION=3.11
FROM ubuntu:24.04 AS base

ARG DATA_FOLDER
ARG EARTHDATA_USERNAME
ARG EARTHDATA_PASSWORD
ARG MODELS_PATH

ENV DATA_FOLDER=${DATA_FOLDER} \
    EARTHDATA_USERNAME=${EARTHDATA_USERNAME} \
    EARTHDATA_PASSWORD=${EARTHDATA_PASSWORD} \
    MODELS_PATH=${MODELS_PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
        libspatialindex-dev \
        libproj-dev \
        gcc \
        g++ \
        make \
        python3-dev \
        python3-pip \
        python3-venv \
        curl \
        git \
        wget \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd -g 1001 instageo-backend && \
    useradd -u 1001 -g instageo-backend --shell /bin/bash --create-home instageo-backend && \
    mkdir -p /app ${DATA_FOLDER} && \
    chown -R instageo-backend:instageo-backend /app ${DATA_FOLDER}

FROM base AS with-uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

FROM with-uv AS with-deps

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV PATH=/app/.venv/bin:$PATH \
    UV_ENV=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app:/app/backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked \
    --extra all \
    --extra cpu \
    --no-install-project \
    --no-editable

FROM with-deps AS final

COPY --chown=instageo-backend:instageo-backend instageo/new_apps/backend /app/backend

WORKDIR /app/backend

USER instageo-backend

RUN echo "MODELS_PATH: ${MODELS_PATH}"

# Create .netrc file for Earthdata access
RUN echo "machine urs.earthdata.nasa.gov login ${EARTHDATA_USERNAME} password ${EARTHDATA_PASSWORD}" > ~/.netrc && \
    chmod 600 ~/.netrc

CMD ["bash"]
