# OpenMedallion Training Environment
# Supports both FinTS forecasting and FinSentiment fine-tuning

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    datasets==2.16.0 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    peft==0.7.1 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    tqdm==4.66.1

# Install time-series specific libraries
RUN pip install --no-cache-dir \
    lightgbm==4.1.0 \
    xgboost==2.0.3 \
    prophet==1.1.5 \
    statsmodels==0.14.1

# Install data collection libraries
RUN pip install --no-cache-dir \
    yfinance==0.2.33 \
    requests==2.31.0 \
    beautifulsoup4==4.12.2

# Install HuggingFace Hub and tracking
RUN pip install --no-cache-dir \
    huggingface-hub==0.20.1 \
    tensorboard==2.15.1 \
    wandb==0.16.1

# Install evaluation and visualization
RUN pip install --no-cache-dir \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.18.0

# Set working directory
WORKDIR /workspace

# Copy package files
COPY setup.py /workspace/
COPY openmedallion/ /workspace/openmedallion/

# Install OpenMedallion package in editable mode
RUN pip install -e /workspace

# Create directories for data and models
RUN mkdir -p /workspace/data \
    /workspace/models \
    /workspace/logs \
    /workspace/outputs

# Set environment variables for HuggingFace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
