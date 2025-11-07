"""Utility functions for BWE model."""

import torch.nn as nn


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights for convolutional layers.

    Args:
        m: Module to initialize
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding to maintain spatial dimensions.

    Args:
        kernel_size: Size of the convolutional kernel
        dilation: Dilation rate

    Returns:
        Padding size
    """
    return int((kernel_size*dilation - dilation)/2)
