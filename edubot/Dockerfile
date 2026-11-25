# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy CUDA packages and keep build fast
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install EduBot dependencies in cached layers for resilience
COPY edubot/requirements.txt /app/edubot/requirements.txt
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    pandas numpy pyyaml scikit-learn pytest pytest-cov ollama
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 chromadb
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 sentence-transformers
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 gradio
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r /app/edubot/requirements.txt

# Pre-download SentenceTransformer and CrossEncoder models into the image
# so the container doesn't download weights upon first student query
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Models are baked into the image; skip HuggingFace network checks on startup
ENV HF_HUB_OFFLINE=1

# Copy datasets and scripts (sibling directory required by edubot)
COPY EduBot_Dataset_v1.2 /app/EduBot_Dataset_v1.2

# Copy edubot application code and vector store
COPY edubot /app/edubot

WORKDIR /app/edubot

# Expose Gradio web UI port
EXPOSE 7860

# Health check to ensure the Gradio server is responding
HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start the EduBot Gradio application
CMD ["python", "src/app.py"]
