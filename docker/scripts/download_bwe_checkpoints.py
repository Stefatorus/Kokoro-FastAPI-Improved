#!/usr/bin/env python3
"""Download AP-BWE checkpoints during Docker build"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


CHECKPOINT_URL = "http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k"
CONFIG_URL = "http://r2.incorpo.ro/static/ai/24k_to_48k/config.json"


def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress bar"""
    try:
        print(f"Downloading from {url}")
        print(f"Saving to {destination}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Stream download with progress bar
        response = requests.get(url, stream=True, allow_redirects=True, timeout=300)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(destination, 'wb') as f:
            if total_size == 0:
                # No content-length header, just download
                f.write(response.content)
                print(f"Downloaded {destination}")
            else:
                # Show progress bar
                print(f"Downloading {total_size / (1024*1024):.1f} MB...")
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)", end='')
                print()  # New line after progress

        print(f"✓ Successfully downloaded to {destination}")
        return True

    except Exception as e:
        print(f"✗ Failed to download from {url}: {e}", file=sys.stderr)
        return False


def main():
    # Checkpoint directory
    checkpoint_dir = Path("/app/AP-BWE/checkpoints/24kto48k")
    checkpoint_path = checkpoint_dir / "g_24kto48k"
    config_path = checkpoint_dir / "config.json"

    print("=" * 60)
    print("AP-BWE Checkpoint Downloader")
    print("=" * 60)

    # Check if already exists
    if checkpoint_path.exists():
        file_size = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Checkpoint already exists ({file_size:.1f} MB)")
        if config_path.exists():
            print(f"✓ Config already exists")
            return 0
    else:
        # Download checkpoint
        print("\nDownloading AP-BWE checkpoint (114 MB)...")
        if not download_file(CHECKPOINT_URL, str(checkpoint_path)):
            return 1

    # Download config if missing
    if not config_path.exists():
        print("\nDownloading config.json...")
        if not download_file(CONFIG_URL, str(config_path)):
            print("Warning: Config download failed, but checkpoint is available")

    print("\n" + "=" * 60)
    print("✓ AP-BWE checkpoints ready!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
