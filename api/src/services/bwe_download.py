"""Automatic download utility for AP-BWE pretrained weights"""

import os
import requests
import zipfile
from pathlib import Path
from loguru import logger
from tqdm import tqdm


# Direct download links for AP-BWE checkpoints
CHECKPOINT_URLS = {
    "24kto48k": {
        "url": "http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k",
        "filename": "g_24kto48k",
        "config_url": "http://r2.incorpo.ro/static/ai/24k_to_48k/config.json",
    }
}


def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress bar

    Args:
        url: URL to download from
        destination: Local path to save file
        chunk_size: Size of chunks to download

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        logger.info(f"Saving to {destination}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Stream download with progress bar
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(destination, 'wb') as f:
            if total_size == 0:
                # No content-length header, just download
                f.write(response.content)
                logger.info(f"Downloaded {destination}")
            else:
                # Show progress bar
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        logger.info(f"Successfully downloaded to {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False


def create_default_config(config_path: str) -> bool:
    """Create default config.json for 24kto48k if it doesn't exist

    Args:
        config_path: Path where config.json should be created

    Returns:
        True if successful
    """
    import json

    config = {
        "num_gpus": 0,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "ConvNeXt_channels": 512,
        "ConvNeXt_layers": 8,
        "segment_size": 8000,
        "n_fft": 1024,
        "hop_size": 80,
        "win_size": 320,
        "hr_sampling_rate": 48000,
        "lr_sampling_rate": 24000,
        "subsampling_rate": 2,
        "num_workers": 4,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        }
    }

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created default config at {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create config: {e}")
        return False


def download_checkpoint(checkpoint_type: str = "24kto48k", base_path: str = None) -> bool:
    """Download AP-BWE checkpoint if it doesn't exist

    Args:
        checkpoint_type: Type of checkpoint (e.g., "24kto48k")
        base_path: Base path for AP-BWE (default: relative to project)

    Returns:
        True if checkpoint is available (already exists or successfully downloaded)
    """
    if checkpoint_type not in CHECKPOINT_URLS:
        logger.error(f"Unknown checkpoint type: {checkpoint_type}")
        return False

    checkpoint_info = CHECKPOINT_URLS[checkpoint_type]

    # Determine base path
    if base_path is None:
        # Try to find AP-BWE directory relative to this file
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent  # Go up to UltraVoiceGen
        base_path = project_root / "AP-BWE"

    checkpoint_dir = Path(base_path) / "checkpoints" / checkpoint_type
    checkpoint_path = checkpoint_dir / checkpoint_info["filename"]
    config_path = checkpoint_dir / "config.json"

    # Check if checkpoint already exists
    if checkpoint_path.exists():
        logger.info(f"Checkpoint already exists at {checkpoint_path}")

        # Check if config exists, download or create if not
        if not config_path.exists():
            if checkpoint_info["config_url"]:
                logger.info(f"Config not found, downloading from {checkpoint_info['config_url']}")
                download_file(checkpoint_info["config_url"], str(config_path))
            else:
                logger.warning(f"Config not found, creating default config")
                create_default_config(str(config_path))

        return True

    # Download checkpoint
    logger.info(f"Checkpoint not found at {checkpoint_path}")
    logger.info(f"Starting automatic download...")

    success = download_file(checkpoint_info["url"], str(checkpoint_path))

    if not success:
        logger.error("Failed to download checkpoint")
        return False

    logger.info(f"Checkpoint downloaded successfully: {checkpoint_path}")

    # Download or create config if it doesn't exist
    if not config_path.exists():
        if checkpoint_info["config_url"]:
            logger.info(f"Downloading config.json from {checkpoint_info['config_url']}")
            config_success = download_file(checkpoint_info["config_url"], str(config_path))
            if not config_success:
                logger.warning("Failed to download config, creating default")
                create_default_config(str(config_path))
        else:
            logger.info("Creating default config.json")
            create_default_config(str(config_path))

    logger.info("✓ Checkpoint download complete!")
    return True


def ensure_checkpoint_available(checkpoint_path: str) -> bool:
    """Ensure checkpoint is available, download if necessary

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        True if checkpoint is available
    """
    checkpoint_file = Path(checkpoint_path)

    # If file exists, we're good
    if checkpoint_file.exists():
        return True

    # Try to determine checkpoint type from path
    if "24kto48k" in str(checkpoint_path):
        checkpoint_type = "24kto48k"
    else:
        logger.error(f"Cannot determine checkpoint type from path: {checkpoint_path}")
        return False

    # Get base path (AP-BWE directory)
    base_path = checkpoint_file.parent.parent.parent

    logger.info(f"Checkpoint not found, attempting automatic download...")
    return download_checkpoint(checkpoint_type, str(base_path))
