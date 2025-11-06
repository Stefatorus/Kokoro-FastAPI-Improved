"""Bandwidth Extension service using AP-BWE"""

import sys
import os
import json
import torch
import torchaudio.functional as aF
import numpy as np
from loguru import logger
from typing import Optional
from pathlib import Path

# Add AP-BWE to path - try multiple possible locations
AP_BWE_PATHS = [
    Path("/app/AP-BWE"),  # Container absolute path
    Path(__file__).parent.parent.parent.parent / "AP-BWE",  # Relative from this file
    Path.cwd() / "AP-BWE",  # From current working directory
]

AP_BWE_PATH = None
for path in AP_BWE_PATHS:
    if path.exists():
        AP_BWE_PATH = path
        sys.path.insert(0, str(AP_BWE_PATH))
        logger.debug(f"Added AP-BWE path: {AP_BWE_PATH}")
        break

if AP_BWE_PATH is None:
    logger.warning(f"AP-BWE directory not found in any of: {[str(p) for p in AP_BWE_PATHS]}")

try:
    from env import AttrDict
    from datasets.dataset import amp_pha_stft, amp_pha_istft
    from models.model import APNet_BWE_Model
    logger.debug("AP-BWE modules imported successfully")
except ImportError as e:
    logger.warning(f"AP-BWE not available: {e}. Please ensure that all requirements in AP-BWE are installed.")
    APNet_BWE_Model = None

# Import download utility
try:
    from .bwe_download import ensure_checkpoint_available
except ImportError as e:
    logger.warning(f"BWE download utility not available: {e}")
    ensure_checkpoint_available = None


class BWEService:
    """Service for bandwidth extension using AP-BWE (24 kHz → 48 kHz)"""

    def __init__(self):
        self.model = None
        self.device = None
        self.config = None
        self._initialized = False
        self._compiled = False

    def initialize(self, checkpoint_path: str, device: Optional[str] = None):
        """Initialize the BWE model

        Args:
            checkpoint_path: Path to the AP-BWE checkpoint file (e.g., 'AP-BWE/checkpoints/24kto48k/g_24kto48k')
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        if APNet_BWE_Model is None:
            logger.error("AP-BWE not available. Please install AP-BWE dependencies.")
            return False

        try:
            # Ensure checkpoint is available (download if necessary)
            if not os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint not found at {checkpoint_path}")
                if ensure_checkpoint_available is not None:
                    logger.info("Attempting automatic download...")
                    if not ensure_checkpoint_available(checkpoint_path):
                        logger.error(f"Failed to download checkpoint")
                        logger.error("Please manually download from:")
                        logger.error("  Checkpoint: http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k")
                        logger.error("  Config: http://r2.incorpo.ro/static/ai/24k_to_48k/config.json")
                        return False
                else:
                    logger.error(f"Checkpoint file not found: {checkpoint_path}")
                    logger.error("Automatic download not available. Please manually download from:")
                    logger.error("  Checkpoint: http://r2.incorpo.ro/static/ai/24k_to_48k/g_24kto48k")
                    logger.error("  Config: http://r2.incorpo.ro/static/ai/24k_to_48k/config.json")
                    return False

            # Load config
            config_file = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
            if not os.path.exists(config_file):
                logger.error(f"Config file not found: {config_file}")
                return False

            with open(config_file) as f:
                json_config = json.loads(f.read())
            self.config = AttrDict(json_config)

            # Set device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            # Load model
            logger.info(f"Loading AP-BWE model from {checkpoint_path}")
            self.model = APNet_BWE_Model(self.config).to(self.device)

            checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_dict['generator'])
            self.model.eval()

            # Optimize model with torch.compile() for faster inference
            # Use 'reduce-overhead' mode for lower latency, 'max-autotune' for throughput
            try:
                if hasattr(torch, 'compile'):
                    logger.info("Compiling BWE model with torch.compile() for optimized inference...")
                    compile_mode = 'reduce-overhead' if str(self.device) == 'cuda' else 'default'
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self._compiled = True
                    logger.info(f"BWE model compiled successfully with mode='{compile_mode}'")
                else:
                    logger.warning("torch.compile() not available (requires PyTorch 2.0+)")
            except Exception as e:
                logger.warning(f"Failed to compile BWE model (will use uncompiled): {e}")
                self._compiled = False

            self._initialized = True
            logger.info(f"AP-BWE initialized successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AP-BWE: {e}")
            return False

    def is_available(self) -> bool:
        """Check if BWE is available and initialized"""
        return self._initialized and self.model is not None

    def enhance(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """Enhance audio from 24 kHz to 48 kHz

        Args:
            audio: Input audio as numpy array (int16 or float32)
                   Shape: (samples,) for mono or (channels, samples) for stereo
            sample_rate: Input sample rate (should be 24000)

        Returns:
            Enhanced audio at 48 kHz as numpy array (int16)
            Always returns mono (samples,) - stereo input is averaged to mono before enhancement

        Note:
            AP-BWE is trained on mono audio. Stereo input will be converted to mono by averaging channels.
        """
        if not self.is_available():
            logger.warning("BWE not initialized, returning original audio")
            return audio

        if sample_rate != 24000:
            logger.warning(f"Expected 24 kHz audio, got {sample_rate} Hz. BWE may not work correctly.")

        try:
            with torch.no_grad():
                # Convert to tensor
                if audio.dtype == np.int16:
                    audio_float = audio.astype(np.float32) / 32768.0
                else:
                    audio_float = audio.astype(np.float32)

                # Check if stereo or mono
                if len(audio_float.shape) == 1:
                    # Mono audio
                    is_stereo = False
                    audio_tensor = torch.FloatTensor(audio_float).unsqueeze(0)
                elif audio_float.shape[0] <= 2 and len(audio_float.shape) == 2:
                    # Stereo audio (channels, samples)
                    is_stereo = True
                    num_channels = audio_float.shape[0]
                    audio_tensor = torch.FloatTensor(audio_float)
                else:
                    # Unexpected shape, treat as mono
                    is_stereo = False
                    audio_tensor = torch.FloatTensor(audio_float).unsqueeze(0)

                if is_stereo:
                    logger.debug(f"Processing stereo audio ({num_channels} channels)")

                    # Average stereo to mono for BWE processing
                    # AP-BWE is trained on mono audio, so we convert to mono
                    logger.debug("Converting stereo to mono for BWE processing")
                    audio_mono = audio_tensor.mean(dim=0, keepdim=True).to(self.device)

                    # STFT at 24kHz - BWE model will extend bandwidth to 48kHz
                    amp_nb, pha_nb, com_nb = amp_pha_stft(
                        audio_mono,
                        self.config.n_fft,
                        self.config.hop_size,
                        self.config.win_size
                    )

                    # Enhance with AP-BWE
                    amp_wb_g, pha_wb_g, com_wb_g = self.model(amp_nb, pha_nb)

                    # ISTFT
                    audio_enhanced = amp_pha_istft(
                        amp_wb_g,
                        pha_wb_g,
                        self.config.n_fft,
                        self.config.hop_size,
                        self.config.win_size
                    )

                    # BWE outputs mono, so we keep it as mono
                    logger.debug("BWE output is mono (from averaged stereo input)")

                else:
                    logger.debug("Processing mono audio")

                    # Move to device
                    audio_tensor = audio_tensor.to(self.device)

                    # STFT at 24kHz - BWE model will extend bandwidth to 48kHz
                    amp_nb, pha_nb, com_nb = amp_pha_stft(
                        audio_tensor,
                        self.config.n_fft,
                        self.config.hop_size,
                        self.config.win_size
                    )

                    # Enhance with AP-BWE
                    amp_wb_g, pha_wb_g, com_wb_g = self.model(amp_nb, pha_nb)

                    # ISTFT
                    audio_enhanced = amp_pha_istft(
                        amp_wb_g,
                        pha_wb_g,
                        self.config.n_fft,
                        self.config.hop_size,
                        self.config.win_size
                    )

                # Convert back to int16
                audio_out = audio_enhanced.squeeze().cpu().numpy()
                audio_out = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)

                return audio_out

        except Exception as e:
            logger.error(f"BWE enhancement failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return audio


# Global BWE service instance
_bwe_service: Optional[BWEService] = None


async def get_bwe_service() -> BWEService:
    """Get or create the global BWE service instance"""
    global _bwe_service
    if _bwe_service is None:
        _bwe_service = BWEService()
    return _bwe_service
