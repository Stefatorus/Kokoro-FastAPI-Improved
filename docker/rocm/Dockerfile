FROM rocm/dev-ubuntu-24.04:6.4.3-complete
ENV DEBIAN_FRONTEND=noninteractive \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install Python and other dependencies
RUN apt-get update && apt upgrade -y && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-ng-data \
    rocrand \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    wget \
    nano \
    g++ \
    cmake \
    make \
    build-essential \
    zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \

    # Install UV using the installer script
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ \

    # Create non-root user and set up directories and permissions
    && useradd -m -u 1001 appuser \
    && mkdir -p /app/api/src/models/v1_0 \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with ROCm extras using cache mount
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1001,gid=1001 \
    uv venv --python 3.12 && \
    uv sync --extra rocm && \
    uv pip install requests tqdm

# Run kbd files
ENV ROCM_VERSION=6.4.3
COPY --chown=appuser:appuser docker/rocm/kbd_install.sh /tmp/
RUN bash /tmp/kbd_install.sh

# Support older GFX Arch
RUN cd /tmp && wget https://archlinux.org/packages/extra/x86_64/rocblas/download -O rocblas.tar.zst \
    && pwd && ls -lah ./ \
    && tar --zstd -xvf rocblas.tar.zst && rm rocblas.tar.zst \
    && rm -rf /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/library/ \
    && mv ./opt/rocm/lib/rocblas/library/ /app/.venv/lib/python3.12/site-packages/torch/lib/rocblas/

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh

# Set all environment variables in one go
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=3 \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="rocm" \
    DOWNLOAD_MODEL=true

# Run FastAPI server through entrypoint.sh
# Model will be downloaded at runtime by entrypoint.sh
CMD ["./entrypoint.sh"]
