# Production Dockerfile (CPU-only, venv-based)

FROM python:3.11-slim AS builder
WORKDIR /app

# Build deps (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (better caching)
COPY requirements.txt .

# Force CPU-only torch/torchaudio (prevents CUDA bloat)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

# Install the rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code
COPY --chown=appuser:appuser . .

# Persistent dirs + HF cache dirs
RUN mkdir -p /opt/models/hf_cache /app/audio_data /app/user_data && \
    chown -R appuser:appuser /opt/models /app

USER appuser

# HF caches on mounted volume
ENV HF_HOME=/opt/models/hf_cache
ENV TRANSFORMERS_CACHE=/opt/models/hf_cache

EXPOSE 8000

# Healthcheck (ensure your API has GET /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker for 8GB RAM instances
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]