FROM ubuntu:latest

# Avoid unnecessary writes to disk
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

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
    && rm -rf /var/lib/{apt,dpkg,cache,log}


# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

# Create a working directory
WORKDIR /app

# Declare build arguments
ARG DATA_FOLDER
ARG EARTHDATA_USERNAME
ARG EARTHDATA_PASSWORD
ARG MODELS_PATH

# Add data folder arg as an environment variable
ENV DATA_FOLDER=${DATA_FOLDER}

# Add earthdata credentials as environment variables
ENV EARTHDATA_USERNAME=${EARTHDATA_USERNAME}
ENV EARTHDATA_PASSWORD=${EARTHDATA_PASSWORD}

# Add models path as an environment variable
ENV MODELS_PATH=${MODELS_PATH}


# Create and activate a virtual environment
ENV VENV_PATH="/opt/instageo-venv"
RUN python3 -m venv $VENV_PATH

# Use the virtualenv for all future commands
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements files first for better caching
# COPY requirements.txt ./
# COPY instageo/new_apps/backend/requirements.txt ./backend-requirements.txt

# Install main requirements
RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # Install backend-specific requirements
# RUN pip install -r backend-requirements.txt

COPY setup.py setup.cfg pyproject.toml ./
COPY instageo ./instageo
RUN pip install -e .[all]

RUN chown -R ubuntu:ubuntu /app
RUN mkdir -p ${DATA_FOLDER} && chown -R ubuntu:ubuntu ${DATA_FOLDER}
USER ubuntu

# Add earthdata credentials in .netrc file
RUN echo machine urs.earthdata.nasa.gov login ${EARTHDATA_USERNAME} password ${EARTHDATA_PASSWORD} >> ~/.netrc

CMD ["bash"]
