# Docker Setup with AP-BWE Integration

This guide explains how to use Kokoro-FastAPI with AP-BWE bandwidth extension in Docker containers.

## Quick Start

### GPU Container with BWE

```bash
# Clone both repositories
git clone https://github.com/Stefatorus/Kokoro-FastAPI-Improved.git
cd Kokoro-FastAPI-Improved
cd ..
git clone https://github.com/yxlu-0102/AP-BWE.git

# Enable BWE in docker-compose
cd Kokoro-FastAPI-Improved/docker/gpu
# Edit docker-compose.yml and set: ENABLE_BWE=true

# Start container
docker compose up --build
```

The checkpoint will **download automatically** on first start (~180 MB).

### CPU Container with BWE

```bash
# Same setup, but use CPU container
cd Kokoro-FastAPI-Improved/docker/cpu
# Edit docker-compose.yml and set: ENABLE_BWE=true

docker compose up --build
```

## Prerequisites

Your directory structure should be:
```
Projects/
├── Kokoro-FastAPI-Improved/
│   ├── api/
│   ├── docker/
│   │   ├── cpu/
│   │   │   ├── Dockerfile
│   │   │   └── docker-compose.yml
│   │   └── gpu/
│   │       ├── Dockerfile
│   │       └── docker-compose.yml
│   └── ...
└── AP-BWE/
    ├── checkpoints/
    │   └── 24kto48k/  # Auto-created on first run
    ├── models/
    └── ...
```

## Configuration

### Enable BWE

Edit the `docker-compose.yml` file in either `docker/gpu/` or `docker/cpu/`:

```yaml
environment:
  # Change this to true to enable BWE
  - ENABLE_BWE=true  # Default: false
  - BWE_CHECKPOINT_PATH=/app/AP-BWE/checkpoints/24kto48k/g_24kto48k
  - BWE_OUTPUT_SAMPLE_RATE=48000
```

### Volume Mounts

Both docker-compose files now include the AP-BWE volume:

```yaml
volumes:
  - ../../api:/app/api
  - ../../../AP-BWE:/app/AP-BWE  # BWE model directory
```

This mounts your local `AP-BWE` directory into the container at `/app/AP-BWE`.

## What Happens on First Start

When you start the container with `ENABLE_BWE=true`:

1. **Dependency Check**: Container verifies BWE dependencies (requests, tqdm)
2. **Checkpoint Detection**: Checks if `/app/AP-BWE/checkpoints/24kto48k/g_24kto48k` exists
3. **Auto-Download**: If missing, downloads from:
   - Checkpoint: `http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k` (~180 MB)
   - Config: `http://r2.incorpo.ro/static/ai/24k_to_48k/config.json` (~1 KB)
4. **Model Load**: Loads AP-BWE model into memory
5. **Ready**: Server starts with BWE enabled!

Example logs:
```
INFO - Checkpoint not found at /app/AP-BWE/checkpoints/24kto48k/g_24kto48k
INFO - Attempting automatic download...
Downloading: 100%|██████████| 180MB/180MB [00:30<00:00, 6.0MB/s]
INFO - ✓ Checkpoint download complete!
INFO - Loading AP-BWE model from /app/AP-BWE/checkpoints/24kto48k/g_24kto48k
INFO - AP-BWE initialized successfully on cuda
```

## Manual Checkpoint Download

If automatic download fails inside the container, you can pre-download on the host:

```bash
# On your host machine (not in container)
mkdir -p AP-BWE/checkpoints/24kto48k

# Download checkpoint
curl -o AP-BWE/checkpoints/24kto48k/g_24kto48k \
  http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k

# Download config
curl -o AP-BWE/checkpoints/24kto48k/config.json \
  http://r2.incorpo.ro/static/ai/24k_to_48k/config.json

# Restart container
cd Kokoro-FastAPI-Improved/docker/gpu  # or /cpu
docker compose restart
```

## Usage Examples

### With Docker Compose

```bash
# Start with BWE enabled
cd docker/gpu  # or docker/cpu
docker compose up

# In another terminal, test the API
curl -X POST "http://localhost:8880/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "This is enhanced 48 kHz audio!",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  --output enhanced_48khz.wav
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="not-needed"
)

# Generate enhanced audio
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    input="High quality bandwidth-extended audio!",
    response_format="wav"
) as response:
    response.stream_to_file("enhanced_output.wav")
```

Check the audio sample rate:
```bash
ffprobe enhanced_output.wav
# Should show: 48000 Hz (with BWE enabled)
# Without BWE: 24000 Hz
```

## Performance Considerations

### GPU Container (Recommended for BWE)
- **Speed**: 292× real-time (BWE adds ~3-4ms per second of audio)
- **Memory**: Base + 200MB for BWE model
- **Quality**: ViSQOL 4.17 (excellent)

### CPU Container
- **Speed**: 18× real-time (BWE adds ~50-60ms per second of audio)
- **Memory**: Base + 200MB for BWE model
- **Quality**: Same as GPU (ViSQOL 4.17)

Both are fast enough for production use!

## Troubleshooting

### Error: "AP-BWE not available"

**Cause**: Dependencies not installed in container

**Solution**: Rebuild the container
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

### Error: "Failed to download checkpoint"

**Cause**: Network issues or firewall blocking download

**Solution 1**: Check container network access
```bash
docker compose exec kokoro-tts curl -I http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k
```

**Solution 2**: Pre-download on host (see "Manual Checkpoint Download" above)

### Error: "Permission denied" when writing checkpoint

**Cause**: Volume mount permissions mismatch

**Solution**: Fix permissions on host
```bash
# Check current ownership
ls -la AP-BWE/checkpoints/

# Fix permissions (GPU uses UID 1001, CPU uses UID 1000)
# For GPU:
sudo chown -R 1001:1001 AP-BWE/checkpoints/
# For CPU:
sudo chown -R 1000:1000 AP-BWE/checkpoints/

# Or make it world-writable (less secure but works)
chmod -R 777 AP-BWE/checkpoints/
```

### BWE Not Working / Still 24 kHz Output

**Check 1**: Verify ENABLE_BWE is set to true
```bash
docker compose exec kokoro-tts env | grep ENABLE_BWE
# Should output: ENABLE_BWE=true
```

**Check 2**: Check container logs for BWE initialization
```bash
docker compose logs | grep BWE
# Should see: "AP-BWE initialized successfully"
```

**Check 3**: Restart container after config changes
```bash
docker compose down
docker compose up
```

## Disabling BWE

To disable bandwidth extension:

1. Edit `docker-compose.yml`:
   ```yaml
   - ENABLE_BWE=false
   ```

2. Restart:
   ```bash
   docker compose restart
   ```

Audio will be generated at 24 kHz without BWE enhancement.

## Container Size Impact

- **Base image**: ~5 GB (CUDA) / ~2 GB (CPU)
- **With BWE dependencies**: +50 MB (requests, tqdm)
- **BWE checkpoint** (in volume): +180 MB
- **Runtime memory**: +200 MB when BWE is enabled

## Advanced Configuration

### Custom Checkpoint Path

```yaml
environment:
  - ENABLE_BWE=true
  - BWE_CHECKPOINT_PATH=/app/AP-BWE/checkpoints/custom/my_checkpoint
```

Make sure the custom checkpoint is accessible via the volume mount.

### Persistent Checkpoint Storage

To avoid re-downloading on container recreate:

```yaml
volumes:
  - ../../api:/app/api
  - ../../../AP-BWE:/app/AP-BWE  # Checkpoints persist here
  # Or use a named volume:
  - bwe_checkpoints:/app/AP-BWE/checkpoints

volumes:
  bwe_checkpoints:
```

### Resource Limits

Limit container resources if needed:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

## Building Custom Images

To build and push custom images with BWE:

```bash
# Build GPU image
cd docker/gpu
docker build -t myregistry/kokoro-bwe-gpu:latest -f Dockerfile ../..
docker push myregistry/kokoro-bwe-gpu:latest

# Build CPU image
cd ../cpu
docker build -t myregistry/kokoro-bwe-cpu:latest -f Dockerfile ../..
docker push myregistry/kokoro-bwe-cpu:latest
```

## Production Deployment

For production with BWE:

1. **Pre-download checkpoints** on host before deployment
2. **Use persistent volumes** for checkpoint storage
3. **Enable health checks** in docker-compose:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8880/docs"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```
4. **Monitor logs** for BWE initialization status
5. **Set resource limits** based on your needs

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Auto-download | ✅ Works | Downloads on first start |
| GPU support | ✅ Works | 292× real-time |
| CPU support | ✅ Works | 18× real-time |
| Volume mounts | ✅ Configured | AP-BWE accessible in container |
| Dependencies | ✅ Installed | requests, tqdm included |
| Persistence | ✅ Works | Checkpoints saved on host |

**Docker + BWE = Production Ready!** 🎉
