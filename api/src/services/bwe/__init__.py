"""Bandwidth Extension (BWE) package - inference only.

This package contains the minimal code needed for AP-BWE inference,
extracted from the original AP-BWE repository to avoid dependency conflicts.
"""

from .config import AttrDict
from .stft import amp_pha_stft, amp_pha_istft
from .model import APNet_BWE_Model, ConvNeXtBlock
from .utils import get_padding, init_weights

__all__ = [
    'AttrDict',
    'amp_pha_stft',
    'amp_pha_istft',
    'APNet_BWE_Model',
    'ConvNeXtBlock',
    'get_padding',
    'init_weights',
]
