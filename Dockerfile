FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      gcc \
      libz-dev \
      libgl1-mesa-glx \
      git \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml setup.py ./
COPY src/ src/
COPY requirements-dev.txt ./

# Install package and dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev]" && \
    pip install -r requirements-dev.txt
