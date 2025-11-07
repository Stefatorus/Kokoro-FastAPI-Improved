FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Install Python and other dependencies
RUN apt-get update -y &&  \
    apt-get install -y python3.10 python3-venv espeak-ng espeak-ng-data git libsndfile1 curl ffmpeg g++ cmake make build-essential &&  \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data &&  \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with GPU extras using cache mount
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1001,gid=1001 \
    uv venv --python 3.10 && \
    uv sync --extra gpu && \
    uv pip install requests tqdm

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser AP-BWE ./AP-BWE
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh

# Download AP-BWE checkpoints during build (114 MB)
# Note: BWE is now integrated into api/src/services/bwe/ - no separate dependencies needed
RUN /app/.venv/bin/python ./download_bwe_checkpoints.py


# Set all environment variables in one go
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api:/app/AP-BWE \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=3 \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="gpu" \
    DOWNLOAD_MODEL=true

# Run FastAPI server through entrypoint.sh
# Model will be downloaded at runtime by entrypoint.sh
CMD ["./entrypoint.sh"]
