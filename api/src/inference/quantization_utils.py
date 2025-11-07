"""
Selective quantization utilities for Kokoro TTS
Implements mixed-precision strategy: keep sensitive parts in FP32/BF16, quantize heavy compute
"""

import torch
import torch.nn as nn
from typing import List, Optional, Set
from loguru import logger


class QuantizationConfig:
    """Configuration for selective quantization."""

    # Presets
    PRESET_BALANCED = "balanced"
    PRESET_MAX_SPEED = "max_speed"
    PRESET_MAX_QUALITY = "max_quality"

    def __init__(self, preset: Optional[str] = None):
        """Initialize with optional preset."""
        # Default: keep everything FP32
        self.encoder_precision = "fp32"
        self.decoder_precision = "fp32"
        self.vocoder_precision = "fp32"
        self.attention_precision = "fp32"
        self.mlp_precision = "fp32"
        self.norm_precision = "fp32"  # Always keep norms in FP32

        if preset:
            self.apply_preset(preset)

    def apply_preset(self, preset: str):
        """Apply a quantization preset."""
        if preset == self.PRESET_BALANCED:
            # Balanced: FP16 for heavy compute, FP32 for sensitive parts
            self.encoder_precision = "fp16"
            self.decoder_precision = "fp16"
            self.vocoder_precision = "bf16"  # Vocoder needs more dynamic range
            self.attention_precision = "fp16"
            self.mlp_precision = "fp16"
            self.norm_precision = "fp32"  # Norms always FP32

        elif preset == self.PRESET_MAX_SPEED:
            # Max speed: FP16 everywhere except vocoder (BF16)
            self.encoder_precision = "fp16"
            self.decoder_precision = "fp16"
            self.vocoder_precision = "bf16"
            self.attention_precision = "fp16"
            self.mlp_precision = "fp16"
            self.norm_precision = "fp32"

        elif preset == self.PRESET_MAX_QUALITY:
            # Max quality: BF16 for most, FP32 for vocoder
            self.encoder_precision = "bf16"
            self.decoder_precision = "bf16"
            self.vocoder_precision = "fp32"
            self.attention_precision = "bf16"
            self.mlp_precision = "bf16"
            self.norm_precision = "fp32"


def apply_precision_to_module(module: nn.Module, precision: str, exclude_norms: bool = True):
    """Apply precision to a module, optionally excluding normalization layers."""
    if precision == "fp32":
        module = module.float()
    elif precision == "fp16":
        module = module.half()
    elif precision == "bf16":
        module = module.to(torch.bfloat16)

    # Keep norms in FP32 for stability
    if exclude_norms:
        for name, submodule in module.named_modules():
            if isinstance(submodule, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                submodule.float()

    return module


def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get a module by its name path."""
    parts = name.split('.')
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def apply_selective_quantization(model: nn.Module, config: QuantizationConfig, module_map: dict):
    """
    Apply selective quantization based on module mapping.

    Args:
        model: The model to quantize
        config: Quantization configuration
        module_map: Dict mapping module categories to module name patterns
                    e.g., {"encoder": ["encoder.layers"], "decoder": ["decoder.layers"]}
    """
    logger.info("Applying selective quantization...")

    # Apply precision to each category
    for category, precision_attr in [
        ("encoder", "encoder_precision"),
        ("decoder", "decoder_precision"),
        ("vocoder", "vocoder_precision"),
        ("attention", "attention_precision"),
        ("mlp", "mlp_precision"),
    ]:
        if category not in module_map:
            continue

        precision = getattr(config, precision_attr)
        patterns = module_map[category]

        logger.info(f"Setting {category} to {precision}")

        for pattern in patterns:
            for name, module in model.named_modules():
                if pattern in name:
                    apply_precision_to_module(module, precision, exclude_norms=True)
                    logger.debug(f"  {name} -> {precision}")

    # Ensure norms are FP32
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.float()
            logger.debug(f"  {name} -> fp32 (norm)")

    return model


# Common module mappings for StyleTTS2-based models
KOKORO_MODULE_MAP = {
    "encoder": [
        "text_encoder",
        "encoder.layers",
    ],
    "decoder": [
        "decoder.layers",
        "acoustic",
    ],
    "vocoder": [
        "decoder.vocoder",
        "istft",
        "hifigan",
        "generator",
    ],
    "attention": [
        ".attn.",
        ".attention.",
        ".self_attn.",
    ],
    "mlp": [
        ".mlp.",
        ".ffn.",
        ".feed_forward.",
    ],
}


def log_model_precision_summary(model: nn.Module):
    """Log a summary of model precision distribution."""
    precision_counts = {
        "fp32": 0,
        "fp16": 0,
        "bf16": 0,
        "other": 0
    }

    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1

        if param.dtype == torch.float32:
            precision_counts["fp32"] += 1
        elif param.dtype == torch.float16:
            precision_counts["fp16"] += 1
        elif param.dtype == torch.bfloat16:
            precision_counts["bf16"] += 1
        else:
            precision_counts["other"] += 1

    logger.info("Model Precision Summary:")
    logger.info(f"  Total parameters: {total_params}")
    for prec, count in precision_counts.items():
        pct = 100 * count / total_params if total_params > 0 else 0
        logger.info(f"  {prec.upper()}: {count} ({pct:.1f}%)")
